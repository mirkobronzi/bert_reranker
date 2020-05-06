import ast
import argparse
import json
import logging
import os
import random

from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


def generate_dataset(data, seed, wrong_answers):

    random.seed(seed)

    all_questions = []
    all_answers = []

    for question, answer_format in zip(data.question, data.answers):
        try:
            if answer_format != "[]":
                all_answers.append(ast.literal_eval(answer_format)[0]["answer"])
                all_questions.append(question)
        except:    # noqa: E722
            continue

    qa_pairs = []
    for idx, question in tqdm(enumerate(all_questions)):
        correct_answer = all_answers[idx]

        candidate_answers = []
        candidate_answers.append(correct_answer)
        random_answers = random.sample(all_answers, wrong_answers)
        while correct_answer in random_answers:
            random_answers = random.sample(all_answers, wrong_answers)
        candidate_answers.extend(random_answers)

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

    seed = 42

    data = pd.read_csv(args.input)
    data = data.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )  # Shuffle the df

    if (args.train_size + args.dev_size + args.test_size) > len(data):
        raise ValueError('train + dev + test size is bigger than data size')

    idx_end_train = args.train_size
    idx_end_dev = idx_end_train + args.dev_size
    idx_end_test = idx_end_dev + args.test_size

    data_train = data.iloc[0:idx_end_train]
    data_dev = data.iloc[idx_end_train:idx_end_dev]
    data_test = data.iloc[idx_end_dev:idx_end_test]

    qa_pairs_train = generate_dataset(data_train, seed, args.wrong_answers)
    qa_pairs_dev = generate_dataset(data_dev, seed, args.wrong_answers)
    qa_pairs_test = generate_dataset(data_test, seed, args.wrong_answers)

    with open(os.path.join(args.output, 'healthtap_train.json'), "w", encoding="utf-8") as ostream:
        json.dump(qa_pairs_train, ostream, indent=4, ensure_ascii=False)

    with open(os.path.join(args.output, 'healthtap_dev.json'), "w", encoding="utf-8") as ostream:
        json.dump(qa_pairs_dev, ostream, indent=4, ensure_ascii=False)

    with open(os.path.join(args.output, 'healthtap_test.json'), "w", encoding="utf-8") as ostream:
        json.dump(qa_pairs_test, ostream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
