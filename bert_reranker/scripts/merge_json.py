import argparse
import json
import logging

from bert_reranker.data.data_loader import get_passages_by_source, get_passage_id

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="emb files to join", required=True, type=str, nargs='+')
    parser.add_argument("--output", help="output file", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    current_pid = 0
    current_example_id = 0
    merged_passages = []
    merged_examples = []

    for input_file in args.inputs:
        pid2cpid = {}

        with open(input_file, 'r', encoding='utf-8') as in_stream:
            input_data = json.load(in_stream)

        _, pid2passages, _ = get_passages_by_source(input_data)

        for example in input_data['examples']:
            example['id'] = current_example_id
            current_example_id += 1

            example_pid = get_passage_id(example)
            if example_pid not in pid2cpid:
                pid2cpid[example_pid] = current_pid
                current_pid += 1

                related_passage = pid2passages[example_pid]
                related_passage['passage_id'] = pid2cpid[example_pid]
                merged_passages.append(example)
            example['passage_id'] = pid2cpid[example_pid]

            merged_examples.append(example)

    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump({'examples': merged_examples, 'passages': merged_passages}, ostream, indent=4,
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
