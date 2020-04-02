import re
import json
import random

from tqdm import tqdm

def remove_html_tags(data):
    p = re.compile(r"<.*?>")
    return p.sub("", data)


def make_qa_pairs_natq(natq_path, n_samples=100):
    with open(natq_path) as json_file:
        contents = json_file.readlines()
        natq = [json.loads(cont) for cont in contents]

    qa_pairs = []
    for ii in range(n_samples):
        question = remove_html_tags(natq[ii]["question"])
        correct_answer = remove_html_tags(natq[ii]["right_paragraphs"][0])
        try:
            wrong_answers = [
                remove_html_tags(natq[ii]["wrong_paragraphs"][0]),
                remove_html_tags(natq[ii]["wrong_paragraphs"][1]),
            ]
        except IndexError:
            # For some reason it can happen that there are not enough
            # wrong paragraphs, so we assume it's insanely unlikely the
            # right paragraph from the previous case would be right again.
            wrong_answers = [
                remove_html_tags(natq[ii]["wrong_paragraphs"][0]),
                remove_html_tags(natq[ii - 1]["right_paragraphs"][0]),
            ]

        candidate_answers = []
        candidate_answers.append(correct_answer)
        candidate_answers.extend(wrong_answers)

        qa_pairs.append([question, candidate_answers])

    return qa_pairs


def make_qa_pairs_faq(faq_path, n_wrong_answers=2, seed=42):
    with open(faq_path, "r") as fh:
        faq = json.load(fh)

    random.seed(seed)
    all_questions = []
    all_answers = []

    for k, v in faq.items():
        if k != "document_URL":
            all_questions.append(k)
            all_answers.append("".join(faq[k]["plaintext"]))

    qa_pairs = []
    for idx, question in enumerate(all_questions):
        correct_answer = all_answers[idx]
        wrong_answers = all_answers.copy()
        wrong_answers.remove(correct_answer)
        random.shuffle(wrong_answers)

        candidate_answers = []
        candidate_answers.append(correct_answer)
        candidate_answers.extend(wrong_answers[:n_wrong_answers])
        qa_pairs.append([question, candidate_answers])

    return qa_pairs


def evaluate_model(ret_trainee, qa_pairs_json_file):

    # Run the test on samples from natq to sanity check evreything is correct
    #  qa_pairs =
    with open(qa_pairs_json_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    correct = 0
    for question, answers in tqdm(qa_pairs):

        out = ret_trainee.retriever.predict(question, answers)

        if out[2][0] == 0:  # answers[0] is always the correct answer
            correct += 1

    acc = correct / len(qa_pairs) * 100
    print("Accuracy: %", acc)


if __name__ == '__main__':

    # Code to generate the faq question-answer pairs
    faq_path = "quebec-en-faq.json"
    n_wrong_answers = 2

    qa_pairs = []

    # Generate 10 rounds of FAQ and correct and wrong answer pairs
    for seed in range(10):
        qa_pairs.extend(
            make_qa_pairs_faq(
                faq_path, n_wrong_answers=n_wrong_answers, seed=seed
            )
        )
    with open('faq_evaluation_set.json', "w", encoding="utf-8") as fp:
        json.dump(qa_pairs, fp, indent=4, ensure_ascii=False)
