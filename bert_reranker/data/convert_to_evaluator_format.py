import argparse
import json
import logging
import random

logger = logging.getLogger(__name__)


def make_qa_pairs_faq(faq_path, n_wrong_answers=2, seed=42):
    with open(faq_path, "r", encoding="utf-8") as fh:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json", required=True)
    parser.add_argument("--output", help="output json", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input, "r", encoding="utf-8") as in_stream:
        qa_pairs = json.load(in_stream)

    question_to_correct_answer_index = {}
    answers_to_index = {}
    index = 0
    for question, answers in qa_pairs:
        for answer in answers:
            if answer not in answers_to_index:
                answers_to_index[answer] = index
                index += 1
        correct_answer = answers[0]
        question_to_correct_answer_index[question] = answers_to_index[correct_answer]

    ordered_answers_index = sorted(answers_to_index.items(), key=lambda item: item[1])
    ordered_answers = [(i, x[0]) for i, x in enumerate(ordered_answers_index)]

    converted = {'questions': question_to_correct_answer_index, 'answers': ordered_answers}
    with open(args.output, "w", encoding="utf-8") as out_stream:
        json.dump(converted, out_stream, indent=4, ensure_ascii=False)

    logger.info('converted {} questions and {} answers'.format(
        len(question_to_correct_answer_index), len(ordered_answers)))


if __name__ == "__main__":
    main()
