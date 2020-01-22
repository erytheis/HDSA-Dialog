import logging
import os
import time
from transformer.Transformer import TransformerDecoder, TableSemanticDecoder
from torch.optim.lr_scheduler import MultiStepLR
from MultiWOZ import get_batch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tools import *
from collections import OrderedDict
from evaluator import evaluateModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = parse_opt()
device = torch.device('cpu')
args.outfile = "/tmp/results.txt.pred.{}".format(args.model)
args.option = "test"
args.model = "BERT_dim128_w_domain"
args.batch_size = 1
args.max_seq_length = 50
args.field = True

with open("{}/vocab.json".format(args.data_dir), 'r') as f:
    vocabulary = json.load(f)

act_ontology = Constants.act_ontology

vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
tokenizer = Tokenizer(vocab, ivocab, False)

logger.info("Loading Vocabulary of {} size".format(tokenizer.vocab_len))
# Loading the dataset

os.makedirs(args.output_dir, exist_ok=True)
checkpoint_file = os.path.join(args.output_dir, args.model)

if 'train' in args.option:
    *train_examples, _ = get_batch(args.data_dir, 'train', tokenizer, args.max_seq_length)
    train_data = TensorDataset(*train_examples)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    *val_examples, val_id = get_batch(args.data_dir, 'val', tokenizer, args.max_seq_length)
    dialogs = json.load(open('{}/val.json'.format(args.data_dir)))
    gt_turns = json.load(open('{}/val_reference.json'.format(args.data_dir)))
elif 'test' in args.option or 'postprocess' in args.option:
    *val_examples, val_id = get_batch(args.data_dir, 'test', tokenizer, args.max_seq_length)
    dialogs = json.load(open('{}/test.json'.format(args.data_dir)))
    if args.non_delex:
        gt_turns = json.load(open('{}/test_reference_nondelex.json'.format(args.data_dir)))
    else:
        gt_turns = json.load(open('{}/test_reference.json'.format(args.data_dir)))

eval_data = TensorDataset(*val_examples)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

BLEU_calc = BLEUScorer()
F1_calc = F1Scorer()

if "BERT" in args.model:
    if args.field:
        decoder = TableSemanticDecoder(vocab_size=tokenizer.vocab_len, d_word_vec=args.emb_dim, n_layers=args.layer_num,
                                       d_model=args.emb_dim, n_head=args.head, dropout=args.dropout)
    elif args.one_hot:
        decoder = TransformerDecoder(vocab_size=tokenizer.vocab_len, d_word_vec=args.emb_dim, act_dim=len(Constants.act_ontology),
                                     n_layers=args.layer_num, d_model=args.emb_dim, n_head=args.head, dropout=args.dropout)
    else:
        decoder = TransformerDecoder(vocab_size=tokenizer.vocab_len, d_word_vec=args.emb_dim, act_dim=Constants.act_len,
                                     n_layers=args.layer_num, d_model=args.emb_dim, n_head=args.head, dropout=args.dropout)
else:
    raise ValueError("Unrecognized Model Type")

decoder.to(device)
loss_func = torch.nn.BCELoss()
loss_func.to(device)

ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=Constants.PAD)
ce_loss_func.to(device)

if args.option == 'train':
    decoder.train()
    if args.resume:
        decoder.load_state_dict(torch.load(checkpoint_file))
        logger.info("Reloaing the encoder and decoder from {}".format(checkpoint_file))

    logger.info("Start Training with {} batches".format(len(train_dataloader)))

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, decoder.parameters()), betas=(0.9, 0.98), eps=1e-09)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)

    best_BLEU = 0
    for epoch in range(360):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, act_vecs, query_results, \
                rep_in, resp_out, belief_state, hierachical_act_vecs, *_ = batch

            decoder.zero_grad()
            optimizer.zero_grad()

            if args.one_hot:
                logits = decoder(tgt_seq=rep_in, src_seq=input_ids, act_vecs=act_vecs)
            else:
                logits = decoder(tgt_seq=rep_in, src_seq=input_ids, act_vecs=hierachical_act_vecs)

            loss = ce_loss_func(logits.contiguous().view(logits.size(0) * logits.size(1), -1).contiguous(),
                                resp_out.contiguous().view(-1))

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                logger.info("epoch {} step {} training loss {}".format(epoch, step, loss.item()))

        scheduler.step()
        if loss.item() < 3.0 and epoch > 0 and epoch % args.evaluate_every == 0:
            logger.info("start evaluating BLEU on validation set")
            decoder.eval()
            # Start Evaluating after each epoch
            model_turns = {}
            TP, TN, FN, FP = 0, 0, 0, 0
            for batch_step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, act_vecs, query_results, \
                    rep_in, resp_out, belief_state, pred_hierachical_act_vecs, *_ = batch

                hyps = decoder.translate_batch(act_vecs=pred_hierachical_act_vecs,
                                               src_seq=input_ids, n_bm=args.beam_size,
                                               max_token_seq_len=40)

                for hyp_step, hyp in enumerate(hyps):
                    pred = tokenizer.convert_id_to_tokens(hyp)
                    file_name = val_id[batch_step * args.batch_size + hyp_step]
                    if file_name not in model_turns:
                        model_turns[file_name] = [pred]
                    else:
                        model_turns[file_name].append(pred)
            BLEU = BLEU_calc.score(model_turns, gt_turns)

            logger.info("{} epoch, Validation BLEU {} ".format(epoch, BLEU))
            if BLEU > best_BLEU:
                torch.save(decoder.state_dict(), checkpoint_file)
                best_BLEU = BLEU
            decoder.train()
elif args.option == "test":
    decoder.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
    logger.info("Loading model from {}".format(checkpoint_file))
    decoder.eval()
    logger.info("Start Testing with {} batches".format(len(eval_dataloader)))

    model_turns = {}
    act_turns = {}
    step = 0
    start_time = time.time()
    TP, TN, FN, FP = 0, 0, 0, 0
    for batch_step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, act_vecs, query_results, \
            rep_in, resp_out, belief_state, pred_hierachical_act_vecs, *_ = batch

        hyps = decoder.translate_batch(act_vecs=pred_hierachical_act_vecs, src_seq=input_ids,
                                       n_bm=args.beam_size, max_token_seq_len=40)
        for hyp_step, hyp in enumerate(hyps):
            pred = tokenizer.convert_id_to_tokens(hyp)
            file_name = val_id[batch_step * args.batch_size + hyp_step]
            if file_name not in model_turns:
                model_turns[file_name] = [pred]
            else:
                model_turns[file_name].append(pred)

            print("INPUT:", tokenizer.convert_id_to_tokens(input_ids.cpu().detach().numpy().T))
            print("ACTS:", convert_act_ids_to_names(pred_hierachical_act_vecs.cpu().detach().flatten().tolist()))
            print("PREDS:", tokenizer.convert_id_to_tokens(hyp))
        logger.info("finished {}/{} used {} sec/per-sent".format(batch_step, len(eval_dataloader),
                                                                 (time.time() - start_time) / args.batch_size))
        start_time = time.time()

    with open(args.outfile + ".pred", 'w') as fp:
        model_turns = OrderedDict(sorted(model_turns.items()))
        json.dump(model_turns, fp, indent=2)

    BLEU = BLEU_calc.score(model_turns, gt_turns)
    entity_F1 = F1_calc.score(model_turns, gt_turns)

    logger.info("BLEU = {} EntityF1 = {}".format(BLEU, entity_F1))
elif args.option == "postprocess":
    with open(args.output_file, 'r') as f:
        model_turns = json.load(f)

    evaluateModel(model_turns)

    success_rate = nondetokenize(model_turns, dialogs)
    BLEU = BLEU_calc.score(model_turns, gt_turns)

    with open('/tmp/results.txt.pred.non_delex', 'w') as f:
        model_turns = OrderedDict(sorted(model_turns.items()))
        json.dump(model_turns, f, indent=2)
    logger.info("Validation BLEU {}, Success Rate {}".format(BLEU, success_rate))

    with open('/tmp/results.txt.non_delex', 'w') as f:
        json.dump(gt_turns, f, indent=2)
else:
    raise ValueError("No such option")
