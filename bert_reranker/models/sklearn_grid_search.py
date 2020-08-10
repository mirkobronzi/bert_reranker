#!/usr/bin/env python

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import GridSearchCV

from bert_reranker.models.sklearn_outliers_model import collect_question_embeddings

logger = logging.getLogger(__name__)

SKLEARN_MODEL_FILE_NAME = "sklearn_outlier_model.pkl"


def add_results_to_df(df, results, fname):
    result_df = pd.DataFrame([results], columns=df.columns)
    df = df.append(result_df)
    df = df.rename(index={0: fname})
    return df


def get_model_and_params(model_name):
    if model_name == "lof":
        base_clf = LocalOutlierFactor()
        parameters = {
            "n_neighbors": [3, 4, 5, 6, 8, 10, 20],
            "contamination": list(np.arange(0.1, 0.5, 0.1)),
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
            "gamma": [0.001, 0.005, 0.01, 0.1],
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
        help="list of embeddings to fine tune the sklearn model on",
        required=True,
    )
    parser.add_argument(
        "--eval-embeddings",
        help="These embeddings will only be evaluated on",
        required=True,
        type=str,
        nargs="+",
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
        embeddings = collect_question_embeddings(args, data)
    else:
        embeddings = []

    if args.train_on_passage_headers:
        passage_header_embs = data["passage_header_embs"]
        embeddings.extend(passage_header_embs)
        logger.info("found {} passage headers embs".format(len(passage_header_embs)))

    logger.info("final size of the collected embeddings: {}".format(len(embeddings)))
    embedding_array = np.concatenate(embeddings)

    # pd dataframe to save results as .csv
    # we save and read to/from disk to bypass the scoring method
    df_columns = [args.test_embeddings]
    df_columns.extend(args.eval_embeddings)
    df_columns.append("contamination")
    df_columns.append("n_neighbors")
    results_df = pd.DataFrame(columns=df_columns)
    results_df.to_csv("results_lof.csv")

    def scoring(estimator, X, y=None, args=args):
        from sklearn.metrics import accuracy_score

        logger.info("\n" * 2)
        logger.info("*" * 50)
        logger.info("sklearn model params {}".format(estimator))

        results_df = pd.read_csv("results_lof.csv", index_col=0)

        # Load testing embeddings for fine tuning
        reported_accuracy = []
        with open(args.test_embeddings, "rb") as in_stream:
            data = pickle.load(in_stream)
        question_embeddings = np.concatenate(data["question_embs"])
        labels = [1 if label == "id" else -1 for label in data["question_labels"]]
        preds = estimator.predict(question_embeddings)
        test_acc = accuracy_score(labels, preds)
        reported_accuracy.append(test_acc)

        logger.info("Evaluating on: {}".format(args.test_embeddings))
        logger.info("Total number of samples: {}".format(len(labels)))
        logger.info("Number of OOD predictions: {}".format((preds == -1).sum()))
        logger.info("Number of ID predictions: {}".format((preds == 1).sum()))
        logger.info("Accuracy: {}".format(test_acc))
        logger.info("=" * 50)

        # Get results on all eval files
        for file in args.eval_embeddings:
            logger.info("Evaluating on: {}".format(file))
            with open(file, "rb") as in_stream:
                data = pickle.load(in_stream)
            question_embeddings = np.concatenate(data["question_embs"])
            if "exclusion" in file:  # if its an exclusion file, everything is OOD
                labels = [-1] * len(data["question_labels"])
            else:
                labels = [
                    1 if label == "id" else -1 for label in data["question_labels"]
                ]
            preds = estimator.predict(question_embeddings)
            acc = accuracy_score(labels, preds)
            logger.info("Total number of samples: {}".format(len(labels)))
            logger.info("Number of OOD predictions: {}".format((preds == -1).sum()))
            logger.info("Number of ID predictions: {}".format((preds == 1).sum()))
            logger.info("Accuracy: {}".format(acc))
            logger.info("=" * 50)

            reported_accuracy.append(acc)

        fname = str(estimator)
        results = [*reported_accuracy, estimator.contamination, estimator.n_neighbors]
        results_df = add_results_to_df(results_df, results, fname)

        results_df.to_csv("results_lof.csv", index=True)
        return test_acc

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


if __name__ == "__main__":
    main()
