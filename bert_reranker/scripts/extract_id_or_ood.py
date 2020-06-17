import argparse
import json
import logging

from bert_reranker.data.data_loader import get_passages_by_source, get_examples, get_passage_id, \
    get_passage_content2pid, get_passage_last_header, is_in_distribution

logger = logging.getLogger(__name__)


def filter_user_questions(input_to_filter, faq_contents):

    with open(input_to_filter, 'r', encoding='utf-8') as in_stream:
        input_data = json.load(in_stream)

    _, pid2passage, _ = get_passages_by_source(input_data, keep_ood=True)

    filtered_example = []
    examples = get_examples(input_data, keep_ood=True)
    for example in examples:
        related_pid = get_passage_id(example)
        related_passage = pid2passage[related_pid]
        if get_passage_last_header(related_passage) in faq_contents:
            filtered_example.append(example)

    logger.info(
        'file {}: passage size {} / pre-filtering example size {} / post filtering examples size'
        ' {}'.format(input_to_filter, len(input_data['passages']), len(examples),
                     len(filtered_example)))

    return {'examples': filtered_example, 'passages': input_data['passages']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="join files to filter", required=True)
    parser.add_argument("--output", help="output file", required=True)
    parser.add_argument("--keep-id", help="will keep id", action="store_true")
    parser.add_argument("--keep-ood", help="will keep ood", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input, 'r', encoding='utf-8') as in_stream:
        input_data = json.load(in_stream)

    filtered = []
    _, pid2passages, _ = get_passages_by_source(input_data)

    for example in get_examples(input_data, True):
        example_pid = get_passage_id(example)
        related_passage = pid2passages[example_pid]
        is_id = is_in_distribution(related_passage)
        if is_id and args.keep_id:
            filtered.append(example)
        elif not is_id and args.keep_ood:
            filtered.append(example)

    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump({'examples': filtered, 'passages': input_data['passages']}, ostream, indent=4,
                  ensure_ascii=False)


def get_faq_contents(faq_file):
    _, faq_pid2passage, _ = get_passages_by_source(faq_file, keep_ood=True)
    faq_passages = list(faq_pid2passage.values())
    source2faq_contents = get_passage_content2pid(faq_passages, duplicates_are_ok=True)

    all_source_contents = []
    for source_contents in source2faq_contents.values():
        all_source_contents.extend(source_contents)
    all_source_contents = set(all_source_contents)

    logger.info('passages in reference file: {}'.format(len(all_source_contents)))
    return all_source_contents


if __name__ == "__main__":
    main()
