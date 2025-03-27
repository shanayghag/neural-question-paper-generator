import torch
import transformers

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42
        self.MODEL_PATH = 't5-base'
        self.TRAINED_MODELS_DIR = 'E:/InC/src/trained_models/'

        # data
        self.SRC_MAX_LENGTH = 300
        self.TGT_MAX_LENGTH = 30
        self.BATCH_SIZE = 8

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 3e-5
        self.OPTIMIZER = 'AdamW'
        # self.CRITERION = 'BCEWithLogitsLoss'
        self.SAVE_BEST_ONLY = True
        self.N_VALIDATE_DUR_TRAIN = 1
        self.EPOCHS = 1

config = Config()