import argparse
import json
import logging
import random

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_dataset(data, seed, n_of_wrong_answers, mode):

    random.seed(seed)

    all_questions = []
    all_gt_questions = []
    all_answers = []

    for question, answer, gt_question in zip(data.question, data.answer, data.gt_question):
        all_answers.append(answer)
        all_questions.append(question)
        all_gt_questions.append(gt_question)

    qa_pairs = []
    for idx in tqdm(range(len(all_questions))):
        if mode == 'qa':
            qa_pair = generate_qa_pair(all_answers, all_questions, idx, n_of_wrong_answers)
        elif mode == 'qq':
            qa_pair = generate_qq_pair(all_gt_questions, all_questions, idx, n_of_wrong_answers)
        else:
            raise ValueError("mode {} not supported".format(mode))
        qa_pairs.append(qa_pair)

    return qa_pairs


def generate_qq_pair(all_gt_questions, all_questions, idx, n_of_wrong_answers):
    cquestion = all_questions[idx]
    correct_gt_question = all_gt_questions[idx]
    candidate_answers = [correct_gt_question]  # first one is always correct
    negative_answers = set(all_gt_questions).copy()
    negative_answers.remove(correct_gt_question)
    negative_answers = list(negative_answers)
    if n_of_wrong_answers > 0:
        negative_answers = random.sample(n_of_wrong_answers)
    candidate_answers.extend(negative_answers)
    qa_pair = [cquestion, candidate_answers]
    return qa_pair


def generate_qa_pair(all_answers, all_questions, idx, n_of_wrong_answers):
    cquestion = all_questions[idx]
    correct_answer = all_answers[idx]
    candidate_answers = [correct_answer]  # first one is always correct
    negative_answers = set(all_answers).copy()
    negative_answers.remove(correct_answer)
    negative_answers = list(negative_answers)
    if n_of_wrong_answers > 0:
        negative_answers = random.sample(n_of_wrong_answers)
    candidate_answers.extend(negative_answers)
    qa_pair = [cquestion, candidate_answers]
    return qa_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input csv file (with healthtap)", required=True)
    parser.add_argument("--output", help="folder where to write the output", required=True)
    parser.add_argument("--wrong-answers", help="how many wrong answers for a given question."
                                                " -1 means to keep all the available ones.",
                        type=int, default=2)
    parser.add_argument("--mode", help="either qa (question to answer) or qq (question to ground"
                                       " truth question)", required=True)
    parser.add_argument("--from-line", help="starts from this line in the csv", type=int, default=0)
    parser.add_argument("--to-line", help="ends at this line in the csv", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = pd.read_csv(args.input)

    if args.to_line == -1:
        to = len(data)
    else:
        to = args.to_line

    data = data[args.from_line: to]

    qa_pairs = generate_dataset(data, 1, args.wrong_answers, args.mode)

    logger.info('generate {} pairs'.format(len(qa_pairs)))

    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump(qa_pairs, ostream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
