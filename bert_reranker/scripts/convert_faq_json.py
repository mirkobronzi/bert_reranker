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
    parser.add_argument("--input", help="input json (in natq format)", required=True, nargs='+')
    parser.add_argument("--output", help="output json", required=True)
    parser.add_argument("--rounds", help="how many times we use the same question", type=int,
                        default=10)
    parser.add_argument("--wrong-answers", help="how many wrong answers for a given question."
                                                " -1 means to keep all the available ones.",
                        type=int, default=2)
    parser.add_argument("--max-size", help="max size for the questions. Default: take all of them",
                        type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    collapsed_json = collapse_jsons(args.input)
    logger.info('collapsed {} files into a single dict with {} elements'.format(
        len(args.input), len(collapsed_json)
    ))

    if args.max_size > 0:
        logger.info('keeping only {} questions'.format(args.max_size))
        collapsed_json = {k: v for k, v in list(collapsed_json.items())[:args.max_size]}

    qa_pairs = []
    for seed in range(args.rounds):
        qa_pairs.extend(
            make_qa_pairs_faq(collapsed_json, n_wrong_answers=args.wrong_answers, seed=seed)
        )
    logger.info('final json contains {} examples'.format(len(qa_pairs)))

    with open(args.output, "w", encoding="utf-8") as out_stream:
        json.dump(qa_pairs, out_stream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
