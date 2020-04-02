import argparse
import json
import logging
import random

from tqdm import tqdm

from bert_reranker.data.data_loader import remove_html_toks

logger = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json (in natq format)", required=True)
    parser.add_argument("--output", help="output json", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Generate 10 rounds of FAQ and correct and wrong answer pairs
    n_wrong_answers = 2
    qa_pairs = []
    for seed in range(10):
        qa_pairs.extend(
            make_qa_pairs_faq(args.input, n_wrong_answers=n_wrong_answers, seed=seed)
        )

    with open(args.output, "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs, out_stream, indent=4)


if __name__ == "__main__":
    main()
