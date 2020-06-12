import argparse
import json
import logging
import random

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_dataset(data, seed, n_of_wrong_answers, mode, candidates):

    random.seed(seed)

    all_questions = []
    all_gt_questions = []
    all_answers = []

    for question, answer, gt_question in zip(data.question, data.answer, data.gt_question):
        all_answers.append(answer)
        all_questions.append(question)
        all_gt_questions.append(gt_question)

    qa_pairs = []
    answer_size = None
    not_found_in_candidates = 0
    for idx in tqdm(range(len(all_questions))):
        if mode == 'qa':
            qa_pair = generate_qa_pair(all_answers, all_questions, idx, n_of_wrong_answers,
                                       candidates)
        elif mode == 'qq':
            qa_pair = generate_qq_pair(all_gt_questions, all_questions, idx, n_of_wrong_answers,
                                       candidates)
            if qa_pair is None:
                not_found_in_candidates += 1
                continue
        else:
            raise ValueError("mode {} not supported".format(mode))
        qa_pairs.append(qa_pair)

        if answer_size is None:
            answer_size = len(qa_pair[1])
        assert len(qa_pair[1]) == answer_size

    if candidates is not None:
        logger.info('not found {} elements in the candidates'.format(not_found_in_candidates))
    logger.info('generate {} pairs - every pair has a pool of {} candidates'.format(
        len(qa_pairs), answer_size))
    return qa_pairs


def generate_qq_pair(all_gt_questions, all_questions, idx, n_of_wrong_answers, candidates):
    cquestion = all_questions[idx]
    correct_gt_question = all_gt_questions[idx]

    if candidates is not None:
        if correct_gt_question not in candidates:
            return None
        all_gt_questions = candidates

    candidate_answers = [correct_gt_question]  # first one is always correct
    negative_answers = set(all_gt_questions).copy()
    negative_answers.remove(correct_gt_question)
    negative_answers = list(negative_answers)
    if n_of_wrong_answers > 0:
        negative_answers = random.sample(n_of_wrong_answers)
    candidate_answers.extend(negative_answers)
    qa_pair = [cquestion, candidate_answers]
    return qa_pair


def generate_qa_pair(all_answers, all_questions, idx, n_of_wrong_answers, candidates):
    cquestion = all_questions[idx]
    correct_answer = all_answers[idx]

    if candidates is not None:
        if correct_answer not in candidates:
            return None
        all_answers = candidates

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
    parser.add_argument("--input", help="input csv file", required=True)
    parser.add_argument("--output", help="folder where to write the output", required=True)
    parser.add_argument("--wrong-answers", help="how many wrong answers for a given question."
                                                " -1 means to keep all the available ones.",
                        type=int, default=2)
    parser.add_argument("--candidates", help="will use this candidates instead of inferring it")
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

    if args.candidates is not None:
        with open(args.candidates, 'r', encoding='utf8') as in_stream:
            candidates = [x.strip() for x in in_stream.readlines()]
    else:
        candidates = None

    data = data[args.from_line: to]
    qa_pairs = generate_dataset(data, 1, args.wrong_answers, args.mode, candidates)

    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump(qa_pairs, ostream, indent=4, ensure_ascii=False)

    logger.info('result written to {}'.format(args.output))


if __name__ == "__main__":
    main()
