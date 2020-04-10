import ast
import argparse
import json
import logging
import random

from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


def generate_dataset(data, seed):

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
        random_answers = random.sample(all_answers, 2)
        while correct_answer in random_answers:
            random_answers = random.sample(all_answers, 2)
        candidate_answers.extend(random_answers)

        qa_pairs.append([question, candidate_answers])

    return qa_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json (in natq format)", required=True)
    parser.add_argument("--output", help="output json", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    seed = 42
    n_dev = 5000
    n_test = 5000

    data = pd.read_csv(args.input)
    data = data.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )  # Shuffle the df

    idx_end_train = len(data) - n_dev - n_test
    idx_end_dev = len(data) - n_test

    data_train = data.iloc[0:idx_end_train]
    data_dev = data.iloc[idx_end_train:idx_end_dev]
    data_test = data.iloc[idx_end_dev:]

    qa_pairs_train = generate_dataset(data_train, seed)
    qa_pairs_dev = generate_dataset(data_dev, seed)
    qa_pairs_test = generate_dataset(data_test, seed)

    with open(args.output + 'healthtap_train.json', "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs_train, out_stream, indent=4, ensure_ascii=False)

    with open(args.output + 'healthtap_train_small.json', "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs_train[0:int(1e5)], out_stream, indent=4, ensure_ascii=False)

    with open(args.output + 'healthtap_train_medium.json', "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs_train[0:int(5e5)], out_stream, indent=4, ensure_ascii=False)

    with open(args.output + 'healthtap_dev.json', "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs_dev, out_stream, indent=4, ensure_ascii=False)

    with open(args.output + 'healthtap_test.json', "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs_test, out_stream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
