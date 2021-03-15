import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 4
EPOCHS = 20
ACC_CUTOFF = 0.75
TRAINING_MODE = 'ba'
BERT_PATH = "bert-base-multilingual-cased"
DATASET_FILE_VERSION_COUNT = 1
MODEL_PATH = f"./inputs/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_{TRAINING_MODE}.bin"
MODEL_PATH_2 = f"./inputs/dump/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_"
TRAINING_FILE = f"./inputs/hindi_sentence_classifier_train_v{DATASET_FILE_VERSION_COUNT}.csv"
if 'cased' in BERT_PATH.split('-'):
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)
else:
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
if 'large' in BERT_PATH.split('-'):
    LINEAR_INPUT_SIZE = 1024
else:
    LINEAR_INPUT_SIZE = 768
