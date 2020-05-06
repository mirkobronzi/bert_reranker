import argparse
import json
import logging

from bert_reranker.data.data_normalization import clean_text

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help='data file containing questions and answers', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.input, 'r', encoding='utf-8') as in_stream:
        qa_pairs = json.load(in_stream)

    for question, answer in qa_pairs:
        cleaned_question = clean_text(question)
        logger.info('{} => {}'.format(question, cleaned_question))


if __name__ == '__main__':
    main()
