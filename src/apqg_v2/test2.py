from app.src.final_run import get_all_ques_papers
from pprint import pprint
import time

start_time = time.time()
chp_path = 'E:/InC/src/apqg_v2/app/data/selfhelp_sample.txt'

with open(chp_path, 'r') as f:
    chapter_text = f.read()

all_ques_papers = get_all_ques_papers(chapter_text)
print(f"Time required to generate all question papers {(start_time-time.time())}")
pprint(all_ques_papers)