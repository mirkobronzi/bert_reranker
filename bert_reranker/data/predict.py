import json
import logging
import math
import pickle

import numpy as np
from tqdm import tqdm

from bert_reranker.data.data_loader import (
    get_passages_by_source,
    _encode_passages,
    get_passage_text, get_question, get_passage_id, is_in_distribution, )

logger = logging.getLogger(__name__)


def get_batched_pairs(qa_pairs, batch_size):
    result = []
    for i in range(0, len(qa_pairs), batch_size):
        result.append(qa_pairs[i: i + batch_size])
    return result


class Predictor:

    def __init__(self, retriever_trainee):
        self.retriever_trainee = retriever_trainee
        self.max_question_len = self.retriever_trainee.retriever.max_question_len
        self.tokenizer = self.retriever_trainee.retriever.tokenizer
        self.retriever = retriever_trainee.retriever

    def generate_predictions(self, json_file, predict_to):

        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        predictions = []
        normalized_scores = []
        indices_of_correct_passage = []
        out_stream = open(predict_to, "w") if predict_to else None

        source2passages, _, passage_id2index = get_passages_by_source(
            json_data
        )
        source2encoded_passages, _, _ = _encode_passages(
            source2passages,
            self.max_question_len,
            self.tokenizer,
        )
        for example in tqdm(json_data["examples"]):
            question = example["question"]
            source = example["source"]
            index_of_correct_passage = passage_id2index[example["passage_id"]]

            prediction, norm_score = self.make_single_prediction(question, source,
                                                                 source2encoded_passages)

            predictions.append(prediction)
            normalized_scores.append(norm_score)
            indices_of_correct_passage.append(index_of_correct_passage)

            if out_stream:
                out_stream.write("-------------------------\n")
                out_stream.write("question:\n\t{}\n".format(question))
                out_stream.write(
                    "prediction: correct? {} / norm score {:3.3} / answer content:"
                    "\n\t{}\n".format(
                        prediction == index_of_correct_passage,
                        norm_score,
                        get_passage_text(source2passages[source][prediction]),
                    )
                )
                out_stream.write(
                    "ground truth:\n\t{}\n\n".format(
                        get_passage_text(source2passages[source][index_of_correct_passage])
                    )
                )

        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            result_message = compute_result_at_threshold(
                predictions, indices_of_correct_passage, normalized_scores, threshold
            )
            logger.info(result_message)
            if out_stream is not None:
                out_stream.write(result_message + "\n")

        if out_stream:
            out_stream.close()

    def make_single_prediction(self, question, source, source2encoded_passages):
        return self.retriever.predict(
            question, source2encoded_passages[source]
        )


class PredictorWithOutlierDetector(Predictor):

    def __init__(self, retriever_trainee, outlier_detector_model):
        super(PredictorWithOutlierDetector, self).__init__(retriever_trainee)
        self.outlier_detector_model = outlier_detector_model

    def make_single_prediction(self, question, source, source2encoded_passages):
        emb_question = self.retriever.embed_question(question)
        in_domain = self.outlier_detector_model.predict(emb_question)
        in_domain = np.squeeze(in_domain)
        if in_domain == 1:  # in-domain
            return self.retriever.predict(
                question, source2encoded_passages[source]
            )
        else:
            return -1, 1.0


def generate_embeddings(ret_trainee, input_file, out_file):
    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    _, pid2passage, _ = get_passages_by_source(json_data)

    embs = []
    labels = []
    for example in tqdm(json_data["examples"]):
        pid = get_passage_id(example)
        passage = pid2passage[pid]
        labels.append('id' if is_in_distribution(passage) else 'ood')
        emb = ret_trainee.retriever.embed_question(get_question(example))
        embs.append(emb)

    to_serialize = {"embeddings": embs, "labels": labels}
    with open(out_file, "wb") as out_stream:
        pickle.dump(to_serialize, out_stream)
    logger.info('saved {} embeddings'.format(len(embs)))


def compute_result_at_threshold(
    predictions, indices_of_correct_passage, normalized_scores, threshold
):
    count = len(indices_of_correct_passage)
    ood_count = sum([x == -1 for x in indices_of_correct_passage])
    id_count = count - ood_count
    correct = 0
    id_correct = 0
    ood_correct = 0

    for i, prediction in enumerate(predictions):
        if normalized_scores[i] >= threshold:
            after_threshold_pred = prediction
            id_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        else:
            after_threshold_pred = -1
            ood_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        correct += int(after_threshold_pred == indices_of_correct_passage[i])
    acc = ((correct / count) * 100) if count > 0 else math.nan
    id_acc = ((id_correct / id_count) * 100) if id_count > 0 else math.nan
    ood_acc = ((ood_correct / ood_count) * 100) if ood_count > 0 else math.nan
    return (
        "threshold {:1.3f}: overall correct: {:3}/{}={:3.2f} - in-distribution correct"
        "{:3}/{}={:3.2f} - out-of-distribution: {:3}/{}={:3.2f}".format(
            threshold,
            correct,
            count,
            acc,
            id_correct,
            id_count,
            id_acc,
            ood_correct,
            ood_count,
            ood_acc,
        )
    )
