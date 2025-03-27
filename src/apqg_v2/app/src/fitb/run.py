from html import entities
import spacy
import pytextrank
from nltk.tokenize import sent_tokenize

import re

from .keyphrase_extractor import KeyphraseExtractors
from .text_ranker import TextRank4Sentences
from copy import deepcopy

class FillInTheBlanksUtils:
    @staticmethod
    def clean_text(text):
        text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
        text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
        text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
        text = re.sub('\t', ' ',  text)
        text = re.sub(r" +", ' ', text)
        return text

    @staticmethod
    def get_text_ranked(text):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        doc = nlp(text)
        for phrase in doc._.phrases:
            print(phrase.text)
            print(phrase.rank, phrase.count)
            print(phrase.chunks)

    @staticmethod
    def gen_fitb_ques_paper(context, number_blanks=1):
        # Count of all sentences
        sentence_tokens = sent_tokenize(context)
        total_sentences = len(sentence_tokens)
        top_k = 0
        if total_sentences > 2:
            top_k = int(total_sentences/2)
        else:
            top_k = 1

        # Get top sentences from the given context using textrank algorithm
        ranker = TextRank4Sentences()
        ranker.analyze(context, sentence_tokens)
        top_ranked_k_sentences = ranker.get_top_sentences(top_k)

        # Using spacy NER to extract entities
        nlp = spacy.load("en_core_web_sm")
        
        # List for storing generated questions
        gen_ftbs = []

        for sent in top_ranked_k_sentences:
            sent = FillInTheBlanksUtils.clean_text(sent)
            keywords = []
            outputs = nlp(sent)
            for word in outputs.ents:
                keywords.append(word.text)
            if len(keywords) == 0:
                keywords = KeyphraseExtractors.text_rank(sent, top_n=number_blanks)
            keywords = keywords[:number_blanks]
            # new_sent = deepcopy(sent)
            # print(keywords)
            for keyword in keywords:
                sent = sent.lower()
                keyword = keyword.lower()
                sent = sent.replace(keyword, "______")
                if sent[0] != "_":
                    sent = sent[0].upper() + sent[1:]
            # print(sent)
            temp = dict()
            temp['question'] = sent
            temp['answer'] = keywords
            temp['marks'] = 0
            gen_ftbs.append(temp)

        return gen_ftbs

        

if __name__ == "__main__":
    text = '''
    Now find the Sulaiman and Kirthar hills to the northwest. Some of the areas where women and men first began to grow crops such as wheat and barley about 8000 years ago are located here. People also began rearing animals like sheep, goat, and cattle, and lived in villages. Locate the Garo hills to the north-east and the Vindhyas in central India. These were some of the other areas where agriculture developed. The places where rice was first grown are to the north of the Vindhyas.
    '''
    ftbs = FillInTheBlanksUtils.generate_fill_in_the_blanks(
        text, 
        number_blanks=2
        )

    print(ftbs)

