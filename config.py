import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACC_CUTOFF = 0.75
LEARNING_RATE = 1e-5
RETRAIN = True
RETRAIN_MODEL_LOC = './inputs/bert-base-multilingual-cased_128_ds1_ba.bin'
TRAINING_MODE = 'ba'
BERT_PATH = "bert-base-multilingual-cased"
DATASET_FILE_VERSION_COUNT = 1
MODEL_PATH = f"./inputs/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_{TRAINING_MODE}_lr{str(LEARNING_RATE)}.bin"
MODEL_PATH_2 = f"./inputs/dump/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_{TRAINING_MODE}_lr{str(LEARNING_RATE)}_"
TRAINING_FILE = f"./inputs/hindi_sentence_classifier_train_v{DATASET_FILE_VERSION_COUNT}.csv"
if 'cased' in BERT_PATH.split('-'):
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)
else:
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
if 'large' in BERT_PATH.split('-'):
    LINEAR_INPUT_SIZE = 1024
else:
    LINEAR_INPUT_SIZE = 768
