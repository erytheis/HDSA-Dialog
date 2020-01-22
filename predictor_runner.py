from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.tokenization import BertTokenizer
import argparse
import os
from os.path import dirname, abspath
from train_predictor import QqpProcessor, convert_examples_to_features, InputExample
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tools import *

output_modes = { 
    "qqp": "classification",
}

do_eval = True
NO_CUDA = True
OUTPUT_MODE = "classification"
MAX_SEQ_LENGTH = 256
LOCAL_RANK = -1
BERT_MODEL = "bert-base-uncased"
DO_LOWER_CASE = True
LOAD_DIR = "checkpoints/predictor/save_step_15120"

ROOT_FILE_PATH = dirname(abspath(__file__))
load_dir = os.path.join(ROOT_FILE_PATH, LOAD_DIR)
device = torch.device('cpu')
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.do_eval = do_eval
args.bert_model = BERT_MODEL
args.no_cuda = NO_CUDA
args.max_seq_length = MAX_SEQ_LENGTH
args.do_lower_case = DO_LOWER_CASE
cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(LOCAL_RANK))

turn_data = {"turn_num" : 0,
             "text_m": "no information",
             "text_a": "conversation start",
             "text_b": "hello good day"}


def boot_model():
    processor = QqpProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    model = BertForSequenceClassification.from_pretrained(load_dir, cache_dir=cache_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    return model, tokenizer, label_list, label_list


def predict(turn_data, model, tokenizer, label_list):
    examples = []
    set_type = "dev"
    file_name = "whatever"
    guid = "%s-%s" % (set_type, file_name)

    label = []
    examples.append(InputExample(file=file_name, turn=turn_data["turn_num"], guid=guid, \
                                 text_m=turn_data["text_m"], text_a=turn_data["text_a"], text_b=turn_data["text_b"], label=label))

    eval_features = convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer, OUTPUT_MODE)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    input_ids = all_input_ids.to(device)
    input_mask = all_input_mask.to(device)
    segment_ids = all_segment_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        logits = torch.sigmoid(logits)
    preds = (logits > 0.4).float()
    preds_numpy = preds.cpu().long().data.numpy()
    preds_strings = convert_act_ids_to_names(preds_numpy[0])
    return preds_strings


model, tokenizer, processor, label_list = boot_model()
preds_strings = predict(turn_data, model, tokenizer, label_list)
print(preds_strings)
