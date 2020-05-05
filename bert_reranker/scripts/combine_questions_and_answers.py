import argparse
import json
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-questions", required=True)
    parser.add_argument("--input-candidates", required=True)
    parser.add_argument("--output", help="output json", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input_candidates, "r", encoding="utf-8") as in_stream:
        candidates = [x. strip() for x in in_stream.readlines()]

    qa_pairs = []
    with open(args.input_questions, "r", encoding="utf-8") as in_stream:
        for question in in_stream:
            qa_pair = [question.strip(), candidates]
            qa_pairs.append(qa_pair)

    with open(args.output, 'w', encoding="utf-8") as out_stream:
        json.dump(qa_pairs, out_stream, indent=4, ensure_ascii=False)

    logger.info('used {} candidates and {} questions'.format(len(candidates), len(qa_pairs)))


if __name__ == "__main__":
    main()
