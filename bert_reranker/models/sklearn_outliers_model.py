#!/usr/bin/env python

import argparse
import logging
import os
import pickle

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

SKLEARN_MODEL_FILE_NAME = "sklearn_outlier_model.pkl"


def get_model_and_params(model_name):
    if model_name == "lof":
        base_clf = LocalOutlierFactor()
        parameters = {
            "n_neighbors": [3, 4, 5, 6],
            "contamination": list(np.arange(0.1, 0.5, 0.05)),
            "novelty": [True],
        }
    elif model_name == "isolation_forest":
        base_clf = IsolationForest()
        parameters = {
            "max_samples": [10, 50, 100, 200, 313],
            "n_estimators": [100, 150, 200],
            "contamination": list(np.arange(0.1, 0.5, 0.1)),
            "max_features": [1, 2, 5],
            "random_state": [42],
        }
    elif model_name == "ocsvm":
        base_clf = OneClassSVM()
        parameters = {
            "kernel": ["linear", "poly", "rbf"],
        }
    elif model_name == "elliptic_env":
        base_clf = EllipticEnvelope()
        parameters = {
            "contamination": list(np.arange(0.1, 0.5, 0.1)),
            "random_state": [42],
        }
    else:
        raise NotImplementedError()

    return base_clf, parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", help="numpy file with embeddings", required=True
    )
    parser.add_argument(
        "--output", help="will store the model output in this folder", required=True
    )
    parser.add_argument(
        "--test-embeddings",
        help="embeddings do evaluate the sklearn model on",
        required=True,
    )
    parser.add_argument(
        "--keep-ood-for-questions",
        help="will keep ood embeddings for questions- by default, they are "
        "filtered out",
        action="store_true",
    )
    parser.add_argument(
        "--train-on-questions",
        help="will include question embeddings in train",
        action="store_true",
    )
    parser.add_argument(
        "--train-on-passage-headers",
        help="will include passage-headers in train",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.embeddings, "rb") as in_stream:
        data = pickle.load(in_stream)

    if args.train_on_questions:
        embeddings, labels = collect_question_embeddings(args, data)
    else:
        embeddings = []

    if args.train_on_passage_headers:
        passage_header_embs = data["passage_header_embs"]
        embeddings.extend(passage_header_embs)
        logger.info("found {} passage headers embs".format(len(passage_header_embs)))
        labels = np.ones(len(embeddings))

    logger.info("final size of the collected embeddings: {}".format(len(embeddings)))
    embedding_array = np.concatenate(embeddings)

    def scoring(estimator, X, y=None, args=args):
        from sklearn.metrics import accuracy_score

        # Load testing embeddings
        with open(args.test_embeddings, "rb") as in_stream:
            data = pickle.load(in_stream)
        question_embeddings = np.concatenate(data["question_embs"])
        labels = [1 if l == "id" else -1 for l in data["question_labels"]]
        preds = estimator.predict(question_embeddings)
        acc = accuracy_score(preds, labels)
        return acc

    models = ["lof", "isolation_forest", "ocsvm", "elliptic_env"]

    best_score = 0

    for model in models:
        logger.info("Testing model: {}".format(model))
        base_clf, parameters = get_model_and_params(model)
        cv = [(slice(None), slice(None))]  # Hack to disable CV
        clf = GridSearchCV(base_clf, parameters, scoring=scoring, cv=cv)
        clf.fit(embedding_array)
        logger.info("best model accuracy: {}".format(clf.best_score_))
        logger.info("best model parameters: {}".format(clf.best_params_))

        if clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_params = clf.best_params_
            best_model_name = model
            logger.info(
                "New overall best model found with accuracy: {}".format(clf.best_score_)
            )
            logger.info("Best model name: {}".format(best_model_name))
            logger.info("Best model params: {}".format(best_params))
            with open(
                os.path.join(args.output, SKLEARN_MODEL_FILE_NAME), "wb"
            ) as out_stream:
                pickle.dump(clf.best_estimator_, out_stream)


def collect_question_embeddings(args, data):
    question_embeddings = data["question_embs"]
    question_labels = data["question_labels"]
    logger.info("found {} question embeddings".format(len(question_embeddings)))
    if not args.keep_ood_for_questions:
        embeddings = []
        labels = []
        for i, embedding in enumerate(question_embeddings):
            if question_labels[i] == "id":
                embeddings.append(embedding)
                labels.append(1)

        logger.info(
            "\t{} question embeddings remain after filtering".format(len(embeddings))
        )
    else:
        embeddings = question_embeddings
        labels = [1 if l == "id" else -1 for l in question_labels]
    return embeddings, labels


if __name__ == "__main__":
    main()
