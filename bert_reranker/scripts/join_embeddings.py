import argparse
import logging
import pickle

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="emb files to join", required=True, type=str, nargs='+')
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if type(args.inputs) != list or len(args.inputs) < 2:
        raise ValueError('please specify at least two input files')

    with open(args.inputs[0], "rb") as in_stream:
        data = pickle.load(in_stream)

    passage_size = len(data['passage_header_embs'])
    logger.info('first data file {} has {} passages / {} examples'.format(
        args.inputs[0], passage_size, len(data['question_embs'])))

    for input_to_merge in args.inputs[1:]:
        with open(input_to_merge, "rb") as in_stream:
            data_to_merge = pickle.load(in_stream)
            data['question_embs'].extend(data_to_merge['question_embs'])
            data['question_labels'].extend(data_to_merge['question_labels'])

            logger.info('data file {} has {} examples'.format(
                input_to_merge, len(data_to_merge['question_embs'])))

            if passage_size != len(data_to_merge['passage_header_embs']):
                logger.warning('file {} has {} passages - while the first file has {}'.format(
                    input_to_merge, len(data_to_merge['passage_header_embs']), passage_size
                ))

    logger.info('final size: {} passages / {} examples'.format(
        len(data['passage_header_embs']), len(data['question_embs'])))

    with open(args.output, "wb") as out_stream:
        pickle.dump(data, out_stream)


if __name__ == "__main__":
    main()
