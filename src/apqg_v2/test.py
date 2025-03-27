from app.src.run import *
from app.src.inference import *
from app.src.config import *
from pprint import pprint

ctxts = {
    0: """
        There are several things we can find out — what people ate, the kinds of clothes they wore, the houses in which they lived. We can find out about the lives of hunters, herders, farmers, rulers, merchants, priests, crafts persons, artists, musicians, and scientists. We can also find out about the games children played, the stories they heard, the plays they saw, the songs they sang.
    """,
    1: """
        You've continued to advertise your intention to change your ways. You've remained in steady contact with the people who matter, regularly reminding them that you're trying to do better. You do this by bringing up your objectives and asking point-blank, "How am I doing?"
    """
}

subj_ques_paper = gen_subj_ques_paper(
    chapter_path='/Users/prithvi/Desktop/Projects/QPG/src/apqg_v2/app/data/history_sample1 copy.txt',
)

pprint(subj_ques_paper)