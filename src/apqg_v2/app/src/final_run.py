import app.src.subjective.run as subj_run
from app.src.mcq.run import MCQGeneratorUtils
from app.src.fitb.run import FillInTheBlanksUtils

def get_all_ques_papers(chapter_text):
    subj_ques_paper, ranked_contexts = subj_run.gen_subj_ques_paper(
        chapter_text=chapter_text,
        chapter_path=None,
    )

    mcq_ques_paper = []
    for rank_idx, ctxt in enumerate(ranked_contexts):
        mcq_questions = MCQGeneratorUtils.gen_mcq_questions(ctxt, rank_idx + 1)
        mcq_ques_paper.extend(mcq_questions)
    print('\n-- MCQ paper generated')

    fitb_ques_paper = FillInTheBlanksUtils.gen_fitb_ques_paper(chapter_text)
    print('\n-- FITB Paper Generated')

    return {
        'subjective': subj_ques_paper,
        'mcq': mcq_ques_paper,
        'fitb': fitb_ques_paper
    }
