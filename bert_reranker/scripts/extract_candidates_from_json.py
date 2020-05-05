import argparse
import json
import logging
import random

logger = logging.getLogger(__name__)


def make_qa_pairs_faq(json_file, n_wrong_answers, seed):

    random.seed(seed)
    all_questions_to_answer_index = {}
    all_answers = []

    answer_size = None

    for k, v in json_file.items():
        if k != "document_URL":
            answer = "".join(json_file[k]["plaintext"])

            if answer not in all_answers:
                all_answers.append(answer)
            answer_index = all_answers.index(answer)
            all_questions_to_answer_index[k] = answer_index

    qa_pairs = []
    for question, answer_index in all_questions_to_answer_index.items():
        correct_answer = all_answers[answer_index]
        wrong_answers = all_answers.copy()
        wrong_answers.remove(correct_answer)
        random.shuffle(wrong_answers)

        candidate_answers = []
        candidate_answers.append(correct_answer)
        if n_wrong_answers > 0:
            wrong_answers = wrong_answers[:n_wrong_answers]

        if answer_size is None:
            answer_size = len(wrong_answers)

        if answer_size != len(wrong_answers):
            raise ValueError('different number of answers')

        candidate_answers.extend(wrong_answers)
        qa_pairs.append([question, candidate_answers])

    logger.info('every question has {} wrong answers'.format(answer_size))
    return qa_pairs


def collapse_jsons(json_files):
    collapsed = {}
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as fh:
            faq = json.load(fh)
            for k, v in faq.items():
                if k != "document_URL":
                    collapsed[k] = v
    return collapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json", required=True)
    parser.add_argument("--output", help="output json", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input, "r", encoding="utf-8") as in_stream:
        qa_pairs = json.load(in_stream)

    all_candidates = set()
    for _, candidates in qa_pairs:
        all_candidates |= set(candidates)

    with open(args.output, "w", encoding="utf-8") as out_stream:
        out_stream.write('\n'.join(all_candidates))

    logger.info('collect {} candidates'.format(len(all_candidates)))


if __name__ == "__main__":
    main()
