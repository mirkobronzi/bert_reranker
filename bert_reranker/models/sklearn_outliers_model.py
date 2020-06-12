#!/usr/bin/env python

import argparse
import logging
import os
import pickle

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)

SKLEARN_MODEL_FILE_NAME = 'sklearn_outlier_model.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', help='numpy file with embeddings', required=True)
    parser.add_argument('--output', help='will store the model output in this folder',
                        required=True)
    parser.add_argument('--keep-ood-for-questions',
                        help='will keep ood embeddings for questions- by default, they are '
                             'filtered out',
                        action='store_true')
    parser.add_argument('--train-on-questions',
                        help='will include question embeddings in train',
                        action='store_true')
    parser.add_argument('--train-on-passage-headers',
                        help='will include passage-headers in train',
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.embeddings, "rb") as in_stream:
        data = pickle.load(in_stream)

    if args.train_on_questions:
        embeddings = collect_question_embeddings(args, data)
    else:
        embeddings = []

    if args.train_on_passage_headers:
        passage_header_embs = data['passage_header_embs']
        embeddings.extend(passage_header_embs)
        logger.info('found {} passage headers embs'.format(len(passage_header_embs)))

    logger.info('final size of the collected embeddings: {}'.format(len(embeddings)))
    embedding_array = np.concatenate(embeddings)

    clf = LocalOutlierFactor(n_neighbors=4, novelty=True, contamination=0.1)
    clf.fit(embedding_array)

    with open(os.path.join(args.output, SKLEARN_MODEL_FILE_NAME), "wb") as out_stream:
        pickle.dump(clf, out_stream)


def collect_question_embeddings(args, data):
    question_embeddings = data['question_embs']
    question_labels = data['question_labels']
    logger.info('found {} question embeddings'.format(len(question_embeddings)))
    if not args.keep_ood_for_questions:
        embeddings = []
        for i, embedding in enumerate(question_embeddings):
            if question_labels[i] == 'id':
                embeddings.append(embedding)
        logger.info('\t{} question embeddings remain after filtering'.format(len(embeddings)))
    else:
        embeddings = question_embeddings
    return embeddings


if __name__ == "__main__":
    main()
