import argparse
import json
import logging

logger = logging.getLogger(__name__)


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
