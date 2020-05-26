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
    parser.add_argument('--keep-ood',
                        help='will keep ood embeddings - by default, they are filtered out',
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.embeddings, "rb") as in_stream:
        embeddings_and_labels = pickle.load(in_stream)

    logger.info('found {} embeddings'.format(len(embeddings_and_labels['embeddings'])))
    if not args.keep_ood:
        embeddings = []
        for i, embedding in enumerate(embeddings_and_labels['embeddings']):
            if embeddings_and_labels['labels'][i] == 'id':
                embeddings.append(embedding)
        logger.info('{} embeddings remain after filtering'.format(len(embeddings)))
    else:
        embeddings = embeddings_and_labels['embeddings']
    embedding_array = np.concatenate(embeddings)

    clf = LocalOutlierFactor(n_neighbors=4, novelty=True)
    clf.fit(embedding_array)

    with open(os.path.join(args.output, SKLEARN_MODEL_FILE_NAME), "wb") as out_stream:
        pickle.dump(clf, out_stream)


if __name__ == "__main__":
    main()
