import argparse
import json
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json", required=True)
    parser.add_argument("--output", help="output json", required=True)
    parser.add_argument("--indices", help="indices of elements to keep", required=True,
                        type=int, nargs='+')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input, "r", encoding="utf-8") as fh:
        json_file = json.load(fh)

    filtered = []
    for i, question in enumerate(json_file):
        if i in args.indices:
            filtered.append(question)

    with open(args.output, "w", encoding="utf-8") as out_stream:
        json.dump(filtered, out_stream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
