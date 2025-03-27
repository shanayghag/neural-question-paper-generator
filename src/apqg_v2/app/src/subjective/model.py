import torch
import transformers
from .config import *

class T5Model(torch.nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()

        self.t5 = transformers.T5ForConditionalGeneration.from_pretrained(config.TRAINED_MODELS_DIR + 't5-base')

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        lm_labels=None
        ):

        return self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )