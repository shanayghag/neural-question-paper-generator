from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from app.src.final_run import get_all_ques_papers

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["GET", "POST"])
def gen_ques_paper():
    if request.method == "POST":
        chapter_text = request.get_json()['chapter']
        if chapter_text:
            all_ques_papers = get_all_ques_papers(chapter_text)
            # dummy = [
            #     {
            #         'question': 'Where is rohit?',
            #         'blooms': 'remember',
            #         'marks': 0,
            #     },
            #     {
            #         'question': 'Who is rohit?',
            #         'blooms': 'understand',
            #         'marks': 0,
            #     },
            #     {
            #         'question': 'Why is rohit?',
            #         'blooms': 'analyse',
            #         'marks': 0,
            #     },
            # ]
            return jsonify(all_ques_papers)
    return {'error': 'Error. Please try again.'}, 400

def start_app():
    app.run(debug=True, host='0.0.0.0')