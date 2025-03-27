'''
Description:
    Script to extract metadata from text.
    For eg. Keywords, Keyphrases are extracted using various algorithms
'''

import string
from pprint import pprint

import pke
from rake_nltk import Rake
from nltk.corpus import stopwords


# Class containing keyphrase extraction algorithms
class KeyphraseExtractors:

    @staticmethod
    def topic_rank(document, top_n=5):
        """
        Description:
            TopicRank is a graph-based model used for keyphrase extraction
            In the case of keyphrase Candidate selection for TopicRank: sequences of nouns and adjectives (i.e. `(Noun|Adj)*`)

        Args:
            document : str
            top_n : Number of keyphrases you want to extract

        Returns:
            keyphrases : list of tuples
        """
        keywords = []
        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(input=document, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(top_n)
        for key in keyphrases:
            keywords.append(key[0])
        return keywords

    @staticmethod
    def text_rank(document,
                  top_n=5,
                  window_size=3,
                  pos={'NOUN', 'PROPN', 'ADJ'},
                  top_percent=0.33):
        """
        Description:
            This model builds a graph that represents the text. 
            A graph based ranking algorithm is then applied to extract the lexical
            units (here the words) that are most important in the text.

        Args:
            document : str
            top_n : int
            window_size : size of sliding window
            pos : set of part-of-speech tags
            top_percent : float

        Returns:
            keyphrases : list of tuples
        """
        keywords = []
        extractor = pke.unsupervised.TextRank()
        extractor.load_document(input=document,
                                language='en',
                                normalization=None)
        extractor.candidate_weighting(
            window=window_size,
            pos=pos,
            top_percent=top_percent)
        keyphrases = extractor.get_n_best(n=top_n)
        for key in keyphrases:
            keywords.append(key[0])
        return keywords

    @staticmethod
    def single_rank(document,
                    top_n=5,
                    window_size=3,
                    pos={'NOUN', 'PROPN', 'ADJ'},
                    top_percent=0.33):
        """
        Description:
            This model is an extension of the TextRank model that uses the number of co-occurrences to weigh edges in the graph.

        Args:
            document : str
            top_n : int
            window_size : size of sliding window
            pos : set of part-of-speech tags
            top_percent : float

        Returns:
            keywords : list
        """
        keywords = []
        extractor = pke.unsupervised.SingleRank()
        extractor.load_document(input=document,
                                language='en',
                                normalization=None)
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(
            window=window_size,
            pos=pos)
        keyphrases = extractor.get_n_best(n=top_n)
        for key in keyphrases:
            keywords.append(key[0])
        return keywords

    @staticmethod
    def position_rank(document,
                      top_n=5,
                      window_size=3,
                      pos={'NOUN', 'PROPN', 'ADJ'},
                      grammar="NP: {<ADJ>*<NOUN|PROPN>+}"):
        """
        Description:
            PositionRank keyphrase extraction model.

        Args:
            document : str
            top_n : int
            window_size : size of sliding window
            pos : set of part-of-speech tags
            grammar : grammar for selecting keyphrase candidates
                eg. grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        Returns:
            keyphrase : list of tuples
        """
        keywords = []
        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(input=document,
                                language='en',
                                normalization=None)
        extractor.candidate_selection(grammar=grammar, maximum_word_number=3)
        extractor.candidate_weighting(window=window_size, pos=pos)
        keyphrases = extractor.get_n_best(n=top_n)
        for key in keyphrases:
            keywords.append(key[0])
        return keywords

    @staticmethod
    def yake_extractor(
            document,
            window=3,
            use_stem=False,
            top_n=5):
        """
        Description:
            YAKE(YET ANOTHER KEYPHRASE EXTRACTOR) keyphrase extraction model

        Args:
            document : str
            window : size of sliding window
            use_stem : bool
            top_n : Number of keyphrases you want to extract

        Returns:
            keyphrases : list
        """
        keywords = []
        extractor = pke.unsupervised.YAKE()
        extractor.load_document(input=document,
                                language='en',
                                normalization=None)
        stoplist = stopwords.words('english')
        extractor.candidate_selection(n=3) #, stoplist=stoplist)
        extractor.candidate_weighting(
            window=window, use_stems=use_stem) #, stoplist=stoplist)
        keyphrases = extractor.get_n_best(n=top_n, threshold=0.8)
        for key in keyphrases:
            keywords.append(key[0])
        return keywords

    @staticmethod
    def rake_extractor(document):
        """
        Description:
            RAKE short for Rapid Automatic Keyword Extraction algorithm,
            is a domain independent keyword extraction algorithm which tries to determine key 
            phrases in a body of text by analyzing the frequency of word appearance and 
            its co-occurance with other words in the text.

        Args:
            doc : str

        Returns:
            keyphrases : list
        """
        extractor = Rake()
        extractor.extract_keywords_from_text(document)
        keyphrases = extractor.get_ranked_phrases()
        return keyphrases

    @staticmethod
    def multipartite_rank(text):
        """
        Description: Extract proper nouns
        """
        keywords = []
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text)
        pos = {'PROPN'}
        #pos = {'VERB', 'ADJ', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos) #, stoplist=stoplist)
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=5)
        for key in keyphrases:
            keywords.append(key[0])
        return keywords


def sanity_check():
    text = "The Nile River fed Egyptian civilization for hundreds of years. It begins near the equator in Africa and flows north to the Mediterranean Sea."
    pprint(text)
    extracted_phrase = KeyphraseExtractors.position_rank(text)
    pprint(extracted_phrase)


if __name__ == "__main__":
    # Testing with one extractor
    sanity_check()
