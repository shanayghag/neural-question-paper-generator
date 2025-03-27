# Imports
from .config import *
from .keyphrase_extractor import KeyphraseExtractors
from .text_ranker import TextRank4Sentences
from .distractor_gen import DistractorUtils

import pandas as pd
import numpy as np

import re
import random
import copy
from tqdm.notebook import tqdm
import gc

import torch
import torch.nn as nn

from nltk import sent_tokenize, word_tokenize
import spacy

from transformers import (
    T5Tokenizer, 
    T5Model,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)

from pprint import pprint

# Setting the device for inference
device = torch.device(DEVICE)

class T5Model(torch.nn.Module):
    def __init__(self, path):
        super(T5Model, self).__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(path)

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        lm_labels=None
        ):

        return self.t5_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

class MCQGeneratorUtils:
    @staticmethod
    def loadmodel():
        model = T5Model(MODELS_DIR + 't5-base')
        # push the model to GPU
        model = model.to(device)
        # load weights of best model
        model.load_state_dict(torch.load(
            f"{MODELS_DIR}mcq_t5.pt", map_location=device))
        return model

    @staticmethod
    def get_question(blooms, ans, ctxt, model):
        src_text = "blooms: %s answer: %s context: %s" % (blooms, ans, ctxt)
        src_tokenized = TOKENIZER.encode_plus(
            src_text, 
            padding="max_length",
            truncation=True,
            max_length=SRC_MAX_LENGTH,
            return_attention_mask=True,
            return_tensors='pt'
        )
        b_src_input_ids = src_tokenized['input_ids'].long().to(device)
        b_src_attention_mask = src_tokenized['attention_mask'].long().to(device)

        model.eval()
        with torch.no_grad():
            # get pred
            pred_ids = model.t5_model.generate(
                input_ids=b_src_input_ids, 
                attention_mask=b_src_attention_mask
            )
            pred_id = pred_ids[0].cpu().numpy()
            pred_decoded = TOKENIZER.decode(pred_id)
            pred_decoded = pred_decoded.replace("<pad> ", "").replace("</s>", "")
            pred_decoded = pred_decoded[0].upper() + pred_decoded[1:]

        return pred_decoded

    @staticmethod
    def gen_mcq_questions(ctxt, rank_idx):
        # Count of all sentences
        sentence_tokens = sent_tokenize(ctxt)
        total_sentences = len(sentence_tokens)
        top_k = 0
        if total_sentences > 2:
            top_k = int(total_sentences/2)
        else:
            top_k = 1

        # Get top sentences from the given context using textrank algorithm
        ranker = TextRank4Sentences()
        ranker.analyze(ctxt, sentence_tokens)
        top_ranked_k_sentences = ranker.get_top_sentences(top_k)
        # print(f'top_ranked_k_sentences: {top_ranked_k_sentences}')

        # Using spacy NER to extract entities
        nlp = spacy.load("en_core_web_sm")
        blooms = ["Remember", "Understand", "Apply", "Analyse", "Create", "Evaluate"]

        # Initializing the model
        model = MCQGeneratorUtils.loadmodel()

        # List to store generated questions
        gen_questions = []
        unique_questions = set()

        for sent in top_ranked_k_sentences:
            keywords = []
            outputs = nlp(sent)
            
            # Getting the keywords
            for word in outputs.ents:
                keywords.append(word.text)
            
            if len(keywords) == 0:
                keywords = KeyphraseExtractors.yake_extractor(sent)

            # print(f'keywords: {keywords}')
            for keyword in keywords:
                try:
                    syn = DistractorUtils.get_wordsense(sent, keyword)
                    if syn is not None:
                        distractors = DistractorUtils.get_distractors_wordnet(syn, keyword)
                        for bloom in blooms:
                            temp_dict = {}
                            question = MCQGeneratorUtils.get_question(bloom, keyword, sent, model)
                            if question not in unique_questions:
                                # Getting the distractors                                
                                # print(keyword)
                                # print(distractors)
                                # print()
                                # Adding to the vector of questions if the number of distractors > 1
                                if len(distractors) != 0:
                                    if len(distractors) >= 3:
                                        keyword = keyword[0].upper() + keyword[1:]
                                        options = distractors[:3]
                                        options.append(keyword)
                                        random.shuffle(options)
                                        unique_questions.add(question)
                                        temp_dict['question'] = question
                                        temp_dict['blooms_level'] = bloom
                                        temp_dict['answer'] = [keyword]
                                        temp_dict['options'] = options
                                        temp_dict['rank'] = rank_idx
                                        temp_dict['marks'] = 0
                                        gen_questions.append(temp_dict)
                except:
                    pass

        return gen_questions

if __name__ == "__main__":
    models_dir = MODELS_DIR
    # blooms = 'remember'
    # ans = "wheat"
    ctxt = '''
    There are several things we can find out — what people ate, the kinds of clothes they wore, the houses in which they lived. We can find out about the lives of hunters, herders, farmers, rulers, merchants, priests, crafts persons, artists, musicians, and scientists. We can also find out about the games children played, the stories they heard, the plays they saw, the songs they sang.

Find the river Narmada on Map 1 (page 2). People have lived along the banks of this river for several hundred thousand years. Some of the earliest people who lived here were skilled gatherers, — that is, people who gathered their food. They knew about the vast wealth of plants in the surrounding forests, and collected roots, fruits and other forest produce for their food. They also hunted animals.

Now find the Sulaiman and Kirthar hills to the northwest. Some of the areas where women and men first began to grow crops such as wheat and barley about 8000 years ago are located here. People also began rearing animals like sheep, goat, and cattle, and lived in villages. Locate the Garo hills to the north-east and the Vindhyas in central India. These were some of the other areas where agriculture developed. The places where rice was first grown are to the north of the Vindhyas.

Trace the river Indus and its tributaries (tributaries are smaller rivers that flow into a larger river). About 4700 years ago, some of the earliest cities flourished on the banks of these rivers. Later, about 2500 years ago, cities developed on the banks of the Ganga and its tributaries, and along the sea coasts.

Locate the Ganga and its tributary called the Son. In ancient times the area along these rivers to the south of the Ganga was known as Magadha now lying in the state of Bihar. Its rulers were very powerful, and set up a large kingdom. Kingdoms were set up in other parts of the country as well.

Throughout, people travelled from one part of the subcontinent to another. The hills and high mountains including the Himalayas, deserts, rivers and seas made journeys dangerous at times, but never impossible. So, men and women moved in search of livelihood, as also to escape from natural disasters like floods or droughts. Sometimes men marched in armies, conquering others’ lands. Besides, merchants travelled with caravans or ships, carrying valuable goods from place to place. And religious teachers walked from village to village, town to town, stopping to offer instruction and advice on the way. Finally, some people perhaps travelled driven by a spirit of adventure, wanting to discover new and exciting places. All these led to the sharing of ideas between people

Look at Map 1 once more. Hills, mountains and seas form the natural frontiers of the subcontinent. While it was difficult to cross these frontiers, those who wanted could and did scale the mountains and cross the seas. People from across the frontiers also came into the subcontinent and settled here. These movements of people enriched our cultural traditions. People have shared new ways of carving stone, composing music, and even cooking food over several hundreds of years.

Two of the words we often use for our country are India and Bharat. The word India comes from the Indus, called Sindhu in Sanskrit. Find Iran and Greece in your atlas. The Iranians and the Greeks who came through the northwest about 2500 years ago and were familiar with the Indus, called it the Hindos or the Indos, and the land to the east of the river was called India. The name Bharata was used for a group of people who lived in the north- west, and who are mentioned in the Rigveda, the earliest composition in Sanskrit (dated to about 3500 years ago). Later it was used for the country.
    '''

    gen_questions = MCQGeneratorUtils.generate_questions(models_dir, ctxt)
    for question in gen_questions:
        pprint(question)
