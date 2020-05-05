import argparse
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input csv", required=True)
    parser.add_argument("--output", help="output folder", required=True)
    parser.add_argument("--group-by", help="will group by this column")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = pd.read_csv(args.input)

    if args.group_by is not None:
        labels = data[args.group_by].unique()
        for label in labels:
            label_data = data.loc[data[args.group_by] == label]
            _write_data(args.output, label, label_data)
    else:
        _write_data(args.output, 'output', data)


def _write_data(output, label, data):
    coutput = os.path.join(output, label) + '.txt'
    with open(coutput, "w", encoding="utf-8") as out_stream:
        for question in data.question:
            out_stream.write(question + '\n')
    logger.info('writing result to {}'.format(coutput))


if __name__ == "__main__":
    main()
