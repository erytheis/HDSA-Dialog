import time
import torch
import transformer.Constants as Constants
import json
import logging
from transformer.Transformer import TableSemanticDecoder
from tools import *
import os
from os.path import dirname, abspath

ROOT_FILE_PATH = dirname(abspath(__file__))
args = parse_opt()

def boot_model():
    path_to_model = os.path.join(ROOT_FILE_PATH, 'checkpoints/generator', args.model)
    path_to_data = os.path.join(ROOT_FILE_PATH, 'data')

    with open("{}/vocab.json".format(path_to_data), 'r') as f:
        vocabulary = json.load(f)
    vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
    device = torch.device('cpu')
    tokenizer = Tokenizer(vocab, ivocab, False)
    decoder = TableSemanticDecoder(vocab_size=tokenizer.vocab_len, d_word_vec=args.emb_dim, n_layers=args.layer_num,
                                   d_model=args.emb_dim, n_head=args.head, dropout=args.dropout)
    decoder.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    return decoder, tokenizer, device

def predict(utterance, acts, model, tokenizer, device):
    utterance = preprocess_utterance(utterance, tokenizer)
    input_ids = torch.LongTensor([tokenizer.convert_tokens_to_ids(utterance)]).to(device)
    act_vector = torch.LongTensor([convert_act_names_to_ids(acts)]).to(device)
    hyps = model.translate_batch(act_vecs=act_vector, src_seq=input_ids,
                               n_bm=args.beam_size, max_token_seq_len=40)
    hyps_tokens = tokenizer.convert_id_to_tokens(hyps[0])
    return hyps_tokens

def preprocess_utterance(utterance, tokenizer):
    return [Constants.SOS_WORD] + tokenizer.tokenize(utterance) + [Constants.EOS_WORD]

