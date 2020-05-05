import argparse
import json
import logging

from tqdm import tqdm

from bert_reranker.data.data_loader import clean_text

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input json (in natq format)', required=True)
    parser.add_argument('--output_train', help='output train json', required=True)
    parser.add_argument('--output_dev', help='output dev json', required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data_dev = []
    data_train = []
    with open(args.input, 'r') as in_stream:
        for line in tqdm(in_stream):
            data = json.loads(line)

            if data['num_positives'] >= 1 and data['num_negatives'] >= 2:
                question = data['question']
                paras = data['right_paragraphs'][:1] + data['wrong_paragraphs'][:2]
                paras = [clean_text(i) for i in paras]
                new_example = [question] + [paras]

                if data['dataset'] == 'train':
                    data_train.append(new_example)

                elif data['dataset'] == 'dev':
                    data_dev.append(new_example)

    with open(args.output_train, 'w') as out_stream:
        json.dump(data_train, out_stream, indent=4)

    with open(args.output_dev, 'w') as out_stream:
        json.dump(data_dev, out_stream, indent=4)


if __name__ == '__main__':
    main()
