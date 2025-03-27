import re
from pprint import pprint

from nltk.tokenize import sent_tokenize
from keyphrase_extractor import KeyphraseExtractors


def generate_fitbs(filename, tokenize: bool):
    """
    Function to generate fill-in-the-blanks
    """
    question_set = {}
    question_set_counter = 0
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            if tokenize:
                sent_tokens = sent_tokenize(line)
                for st in sent_tokens:
                    multipartite_res = KeyphraseExtractors.multipartite_rank(
                        st)
                    for keyword in multipartite_res:
                        temp_dict = {}
                        replace_str_len = len(keyword)
                        replace_str = "_" * replace_str_len
                        temp_keyword = r'\b' + keyword + r'\b'
                        temp_dict["question"] = re.sub(
                            temp_keyword, replace_str, st, flags=re.I)
                        temp_dict["answer"] = keyword
                        question_set_counter += 1
                        question_set[question_set_counter] = temp_dict
            else:
                multipartite_res = KeyphraseExtractors.multipartite_rank(
                    line)
                for keyword in multipartite_res:
                    temp_dict = {}
                    replace_str_len = len(keyword)
                    replace_str = "_" * replace_str_len
                    temp_keyword = r'\b' + keyword + r'\b'
                    temp_dict["question"] = re.sub(
                        temp_keyword, replace_str, line, flags=re.I)
                    temp_dict["answer"] = keyword
                    question_set_counter += 1
                    question_set[question_set_counter] = temp_dict

    return question_set


if __name__ == "__main__":
    filename = "E:/CL/mcqs-gen/fill-in-the-blanks/chapter.txt"
    question_set = generate_fitbs(filename, tokenize=True)
    pprint(question_set)
