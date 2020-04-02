import argparse
import json
import logging

from tqdm import tqdm

from bert_reranker.data.data_loader import remove_html_toks

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help='input json (in natq format)', required=True)
    parser.add_argument('--output', help='output json', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = []
    with open(args.input, 'r') as in_stream:
        for line in tqdm(in_stream):
            data = json.loads(line)

            if data['num_positives'] >= 1 and data['num_negatives'] >= 2:
                if data['dataset'] == 'dev':
                    question = data['question']
                    paras = data['right_paragraphs'][:1] + data['wrong_paragraphs'][:2]
                    paras = [remove_html_toks(i) for i in paras]
                    new_example = [question] + [paras]
                    result.append(new_example)

    with open(args.output, 'w') as out_stream:
        json.dump(result, out_stream, indent=4)


if __name__ == '__main__':
    main()

