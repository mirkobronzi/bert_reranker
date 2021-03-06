"""
Script used compare two prediction files.
"""
import argparse
import logging
import re
from collections import defaultdict
from pprint import pformat

PREDICTION_RE = re.compile('^prediction: ')
END_PREDICTION_RE = re.compile(' .*$')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input1", help="first prediction file", required=True)
    parser.add_argument("--input2", help="second prediction file", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input1, 'r', encoding='utf-8') as in_stream:
        lines1 = in_stream.readlines()

    with open(args.input2, 'r', encoding='utf-8') as in_stream:
        lines2 = in_stream.readlines()

    result1 = parse_results(lines1)
    result2 = parse_results(lines2)

    analyze(result1, result2)


def parse_results(lines):
    result = []
    next_line_is_question = False
    next_line_is_prediction = False
    question = None
    for line in lines:
        if next_line_is_question:
            question = line.strip()
            next_line_is_prediction = True
            next_line_is_question = False
        elif line.strip() == 'question:':
            next_line_is_question = True
        elif line.startswith('prediction:'):
            assert next_line_is_prediction
            prediction = PREDICTION_RE.sub("", line)
            prediction = END_PREDICTION_RE.sub("", prediction).strip()
            result.append((question, prediction))
            next_line_is_prediction = False
            next_line_is_question = False
    return result


def analyze(result1, result2):
    if len(result1) != len(result2):
        raise ValueError('parsed results have different sizes')

    differences = []
    same = defaultdict(int)
    different = defaultdict(int)
    for i in range(len(result1)):
        if result1[i][1] != result2[i][1]:
            differences.append((result1[i], result2[i]))
            different[result1[i][1] + '<>' + result2[i][1]] += 1
        else:
            same[result1[i][1]] += 1

    print('summary:\nsame results: {}\n'.format(pformat(same)))
    print('summary:\ndifferent results: {}\n'.format(pformat(different)))
    differences_text = {k: "" for k in different.keys()}
    for left, right in differences:
        label = left[1] + '<>' + right[1]
        assert label in differences_text.keys()
        example_text = '<<\n' + left[0] + '\n' + left[1] + '\n--\n' + right[0] + '\n' + right[1] + \
                       '\n>>\n'
        differences_text[label] += example_text

    print('\n\ndifferences:')
    for label, text in differences_text.items():
        print("***: {}".format(label))
        print(text)


if __name__ == "__main__":
    main()
