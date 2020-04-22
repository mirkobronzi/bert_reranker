import argparse
import json
import logging
import random

logger = logging.getLogger(__name__)


def make_qa_pairs_faq(json_file, n_wrong_answers, seed):

    random.seed(seed)
    all_questions = []
    all_answers = []

    for k, v in json_file.items():
        if k != "document_URL":
            all_questions.append(k)
            all_answers.append("".join(json_file[k]["plaintext"]))

    qa_pairs = []
    for idx, question in enumerate(all_questions):
        correct_answer = all_answers[idx]
        wrong_answers = all_answers.copy()
        wrong_answers.remove(correct_answer)
        random.shuffle(wrong_answers)

        candidate_answers = []
        candidate_answers.append(correct_answer)
        if n_wrong_answers > 0:
            wrong_answers = wrong_answers[:n_wrong_answers]
        candidate_answers.extend(wrong_answers)
        qa_pairs.append([question, candidate_answers])

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    collapsed_json = collapse_jsons(args.input)
    logger.info('collapsed {} files into a single dict with {} elements'.format(
        len(args.input), len(collapsed_json)
    ))

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
