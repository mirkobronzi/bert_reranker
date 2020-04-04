import ast
import argparse
import json
import logging
import random

from tqdm import tqdm
import pandas as pd

from bert_reranker.data.data_loader import remove_html_toks

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json (in natq format)", required=True)
    parser.add_argument("--output", help="output json", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    n_examples = 4000
    seed = 42
    n_wrong_answers = 2

    data = pd.read_csv(args.input)
    data = data.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )  # Shuffle the df
    data = data.iloc[0:n_examples]  # Extract only a subset

    all_questions = []
    all_answers = []

    for question, answer_format in zip(data.question, data.answers):
        if answer_format != "[]":
            all_questions.append(question)
            all_answers.append(ast.literal_eval(answer_format)[0]["answer"])

    random.seed(seed)
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

    with open(args.output, "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs, out_stream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
