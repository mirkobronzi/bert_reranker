import ast
import argparse
import json
import logging
import os
import random

from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


def generate_dataset(data, seed, n_of_wrong_answers):

    random.seed(seed)

    all_questions = []
    all_answers = []

    for question, answer in zip(data.question, data.answer):
        all_answers.append(answer)
        all_questions.append(question)

    qa_pairs = []
    for idx, question in tqdm(enumerate(all_questions)):
        correct_answer = all_answers[idx]

        candidate_answers = [correct_answer]  # first one is always correct

        negative_answers = set(all_answers).copy()
        negative_answers.remove(correct_answer)
        negative_answers = list(negative_answers)
        if n_of_wrong_answers > 0:
            negative_answers = random.sample(n_of_wrong_answers)

        candidate_answers.extend(negative_answers)
        qa_pairs.append([question, candidate_answers])

    return qa_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input csv file (with healthtap)", required=True)
    parser.add_argument("--output", help="folder where to write the output", required=True)
    parser.add_argument("--wrong-answers", help="how many wrong answers for a given question."
                                                " -1 means to keep all the available ones.",
                        type=int, default=2)
    parser.add_argument("--train-size", type=int, default=1500000)
    parser.add_argument("--dev-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = pd.read_csv(args.input)
    qa_pairs_train = generate_dataset(data, 1, args.wrong_answers)
    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump(qa_pairs_train, ostream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
