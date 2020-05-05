import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input csv", required=True)
    parser.add_argument("--output", help="output txt", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = pd.read_csv(args.input)

    with open(args.output, "w", encoding="utf-8") as out_stream:
        for question in data.question:
            out_stream.write(question + '\n')


if __name__ == "__main__":
    main()
