# Configuration for LLMs
import torch
from transformers import T5Tokenizer

SEED = 42
MODEL_PATH = 't5-base'

# Trained model
MODELS_DIR = 'E:/InC/src/trained_models/'
# T5_MODEL_PATH = "E:/CL/mcqs-gen/4-choice-mcq/trained models/t5-base/"

# data
TOKENIZER = T5Tokenizer.from_pretrained(MODELS_DIR + 't5-base')
SRC_MAX_LENGTH = 640
TGT_MAX_LENGTH = 96
BATCH_SIZE = 6

# model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FULL_FINETUNING = True
LR = 3e-5
OPTIMIZER = 'AdamW'
# CRITERION = 'BCEWithLogitsLoss'
SAVE_BEST_ONLY = True
N_VALIDATE_DUR_TRAIN = 0
EPOCHS = 2