import time
import torch
from .config import *
from .model import *
from .inference import *
from pprint import pprint

from .text_ranker import TextRank4Sentences

def get_contexts(chapter_text=None, chapter_path=None):
    contexts = []

    if chapter_path:
        with open(chapter_path, 'r') as f:
            chapter_text = f.read()

    contexts_li = chapter_text.split("\n\n")
    for item in contexts_li:
        contexts.append(item.replace('\n', '').replace('\t',''))

    return contexts

def rank(contexts):
    txtrnk = TextRank4Sentences()
    txtrnk.analyze(contexts)
    ranked_contexts = txtrnk.get_top_sentences(len(contexts))
    return ranked_contexts

def get_model():
    device = config.DEVICE
    model = T5Model().to(device)
    model_state_dict = torch.load(
        config.TRAINED_MODELS_DIR + 'subj_t5_v2.pt',
        map_location=device,
    )
    model.load_state_dict(model_state_dict)
    return model

def get_all_questions(model, ranked_contexts):
    all_questions = dict()
    blooms_levels = ["Remember", "Understand", "Apply", "Analyse", "Create", "Evaluate"]

    uniq = set()
    for ctxt_idx, ctxt in enumerate(ranked_contexts):
        questions = dict()
        for blvl in blooms_levels:
            ques = get_question(model, blvl, ctxt)
            if ques not in uniq:
                uniq.add(ques)
                questions[blvl] = ques[6].upper() + ques[7:-4] + '?'
        all_questions[ctxt_idx] = questions

    return all_questions

def gen_subj_ques_paper(
    chapter_text=None, 
    chapter_path=None
    ):
    contexts = get_contexts(chapter_text, chapter_path)
    print("\n-- Contexts obtained")

    ranked_contexts = rank(contexts)
    print('\n-- Contexts ranked')
    # print(f'\nori contexts: {contexts}')
    # print(f'ranked contexts: {ranked_contexts}')

    model = get_model()
    print("\n-- Model loaded")

    all_questions = get_all_questions(model, ranked_contexts)
    print("\n-- Questions generated")

    subj_ques_paper = []
    for idx in all_questions:
        for blvl, ques in all_questions[idx].items():
            subj_ques_paper.append({
                'question': ques,
                'blooms_level': blvl,
                'answer': None,
                'options': None,
                'rank': idx + 1,
                'marks': 0,
            })
    print('\n-- Subjective question paper generated\n')

    return subj_ques_paper, ranked_contexts