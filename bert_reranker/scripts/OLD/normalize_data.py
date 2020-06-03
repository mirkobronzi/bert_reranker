import argparse
import json
import logging
import re

import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)


def remove_links(s):
    """swap a link with a _URL_ token"""
    return re.sub(r"http\S+", "_URL_", s)


def remove_html_tags(s):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', s)


NON_STANDARD_PUNCTUATION_RE = re.compile('¿')


def remove_non_standard_punctuation(s):
    return NON_STANDARD_PUNCTUATION_RE.sub(' ', s)


def remove_extra_whitespace(s):
    s = ' '.join(s.split())
    return s


def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sentence)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    filtered_sentence = ' '.join(filtered_tokens)
    return filtered_sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help='data file containing questions and answers', required=True)
    parser.add_argument("--output", help="json file where to write the output", required=True)
    parser.add_argument("--remove-urls", action='store_true')
    parser.add_argument("--remove-stopwords", action='store_true')
    parser.add_argument("--remove-non-standard-punctuation", action='store_true')
    parser.add_argument("--strip", help='removes leading/trailing spaces', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.input, 'r', encoding='utf-8') as in_stream:
        qa_pairs = json.load(in_stream)

    cleaned_qa_pqirs = []
    for question, answers in tqdm.tqdm(qa_pairs):
        if args.remove_urls:
            answers = [remove_links(answer) for answer in answers]
            question = remove_links(question)
        if args.remove_stopwords:
            answers = [remove_stopwords(answer) for answer in answers]
            question = remove_stopwords(question)
        if args.remove_non_standard_punctuation:
            answers = [remove_non_standard_punctuation(answer) for answer in answers]
            question = remove_non_standard_punctuation(question)
        if args.strip:
            answers = [answer.strip() for answer in answers]
            question = question.strip()
        cleaned_qa_pqirs.append((question, answers))

    logger.info('final json contains {} examples'.format(len(qa_pairs)))
    with open(args.output, "w", encoding="utf-8") as out_stream:
        json.dump(cleaned_qa_pqirs, out_stream, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
