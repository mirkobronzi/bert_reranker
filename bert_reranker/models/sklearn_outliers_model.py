#!/usr/bin/env python

import argparse
import logging
import pickle

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', help='numpy file with embeddings', required=True)
    parser.add_argument('--output', help='will store the model output here', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.embeddings, "rb") as in_stream:
        questions_and_labels = pickle.load(in_stream)

    clf = LocalOutlierFactor(n_neighbors=4, novelty=True)

    questions = np.concatenate(questions_and_labels['questions'])
    clf.fit(questions)
    with open(args.output, "wb") as out_stream:
        pickle.dump(clf, out_stream)


if __name__ == "__main__":
    main()
