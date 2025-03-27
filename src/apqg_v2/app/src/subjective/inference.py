import torch
from .config import *
import transformers

def get_question(model, blooms, ctxt):
    device = config.DEVICE

    # src_text = "blooms: %s  context: %s </s>" % (blooms, ctxt)
    src_text = f'blooms: {blooms.lower()}  context: {ctxt}'
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        config.TRAINED_MODELS_DIR + 't5-base'
    )
    src_tokenized = tokenizer.encode_plus(
        src_text, 
        padding="max_length",
        truncation=True,
        max_length=config.SRC_MAX_LENGTH,
        return_attention_mask=True,
        return_tensors='pt'
    )
    b_src_input_ids = src_tokenized['input_ids'].long().to(device)
    b_src_attention_mask = src_tokenized['attention_mask'].long().to(device)

    model.eval()
    with torch.no_grad():
        # get pred
        pred_ids = model.t5.generate(
            input_ids=b_src_input_ids, 
            attention_mask=b_src_attention_mask
        )
        pred_id = pred_ids[0].cpu().numpy()
        pred_decoded = tokenizer.decode(pred_id)
        
    return pred_decoded