import argparse
import json
import logging
import ntpath
import os

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-questions", help='files with one question per line',
                        required=True, nargs='+')
    parser.add_argument("--input-candidates", help='file with one candidate per line',
                        required=True)
    parser.add_argument("--output", help="output folder", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input_candidates, "r", encoding="utf-8") as in_stream:
        candidates = [x. strip() for x in in_stream.readlines()]

    for input_question in args.input_questions:
        _generate_json(
            input_question,
            os.path.join(args.output, ntpath.basename(input_question)) + '.json', candidates)


def _generate_json(input_file, output_file, candidates):
    qa_pairs = []
    with open(input_file, "r", encoding="utf-8") as in_stream:
        for question in in_stream:
            qa_pair = [question.strip(), candidates]
            qa_pairs.append(qa_pair)
    with open(output_file, 'w', encoding="utf-8") as out_stream:
        json.dump(qa_pairs, out_stream, indent=4, ensure_ascii=False)
    logger.info('file {}: used {} candidates and {} questions'.format(
        output_file, len(candidates), len(qa_pairs)))


if __name__ == "__main__":
    main()
