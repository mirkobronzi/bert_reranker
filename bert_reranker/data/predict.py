import json
import logging
import math
import pickle

import numpy as np
from tqdm import tqdm

from bert_reranker.data.data_loader import (
    get_passages_by_source,
    _encode_passages,
    get_passage_last_header, get_question, get_passage_id, is_in_distribution, )

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
        self.no_candidate_warnings = 0

    def generate_predictions(self, json_file, predict_to, multiple_thresholds):

        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        predictions = []
        questions = []
        sources = []
        normalized_scores = []
        indices_of_correct_passage = []

        source2passages, _, passage_id2index = get_passages_by_source(
            json_data
        )
        source2encoded_passages, _, _ = _encode_passages(
            source2passages,
            self.max_question_len,
            self.tokenizer,
        )

        self.compute_results(indices_of_correct_passage, json_data, normalized_scores,
                             passage_id2index, predictions, questions, source2encoded_passages,
                             sources)
        generate_and_log_results(indices_of_correct_passage, normalized_scores, predict_to,
                                 predictions, questions, source2passages, sources,
                                 multiple_thresholds=multiple_thresholds)

    def compute_results(self, indices_of_correct_passage, json_data, normalized_scores,
                        passage_id2index, predictions, questions, source2encoded_passages, sources):
        for example in tqdm(json_data["examples"]):
            question = example["question"]
            questions.append(question)
            source = example["source"]
            sources.append(source)
            index_of_correct_passage = passage_id2index[example["passage_id"]]

            prediction, norm_score = self.make_single_prediction(question, source,
                                                                 source2encoded_passages)

            predictions.append(prediction)
            normalized_scores.append(norm_score)
            indices_of_correct_passage.append(index_of_correct_passage)

    def make_single_prediction(self, question, source, source2encoded_passages):
        candidates = source2encoded_passages[source]
        if candidates:
            return self.retriever.predict(question, candidates)
        else:
            self.no_candidate_warnings += 1
            logger.warning('no candidates for source {} - returning 0 by default (so far, this '
                           'happened {} times)'.format(source, self.no_candidate_warnings))
            return 0, 1.0


class PredictorWithOutlierDetector(Predictor):

    def __init__(self, retriever_trainee, outlier_detector_model):
        super(PredictorWithOutlierDetector, self).__init__(retriever_trainee)
        self.outlier_detector_model = outlier_detector_model

    def make_single_prediction(self, question, source, source2encoded_passages):
        emb_question = self.retriever.embed_question(question)
        in_domain = self.outlier_detector_model.predict(emb_question)
        in_domain = np.squeeze(in_domain)
        if in_domain == 1:  # in-domain
            return super(PredictorWithOutlierDetector, self).make_single_prediction(
                question, source, source2encoded_passages)
        else:
            return -1, 1.0


def generate_and_log_results(indices_of_correct_passage, normalized_scores, predict_to,
                             predictions, questions, source2passages, sources,
                             multiple_thresholds=False):
    with open(predict_to, "w") as out_stream:
        log_results_to_file(indices_of_correct_passage, normalized_scores, out_stream,
                            predictions, questions, source2passages, sources)

        out_stream.write('results:\n\n')
        if multiple_thresholds:
            for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                result_message = compute_result_at_threshold(
                    predictions, indices_of_correct_passage, normalized_scores, threshold, True
                )
                logger.info(result_message)
                out_stream.write(result_message + "\n")
        else:
            result_message = compute_result_at_threshold(
                predictions, indices_of_correct_passage, normalized_scores, 0.0, False
            )
            logger.info(result_message)
            out_stream.write(result_message + "\n")


def log_results_to_file(indices_of_correct_passage, normalized_scores, out_stream,
                        predictions, questions, source2passages, sources):
    for i in range(len(predictions)):
        question = questions[i]
        prediction = predictions[i]
        index_of_correct_passage = indices_of_correct_passage[i]
        norm_score = normalized_scores[i]
        source = sources[i]
        out_stream.write("-------------------------\n")
        out_stream.write("question:\n\t{}\n".format(question))

        if prediction == index_of_correct_passage and prediction == -1:
            pred_outcome = "OOD_CORRECT"
        elif prediction == index_of_correct_passage and prediction >= 0:
            pred_outcome = "ID_CORRECT"
        elif prediction != index_of_correct_passage and index_of_correct_passage == -1:
            pred_outcome = "OOD_MISCLASSIFIED_AS_ID"
        elif prediction == -1 and index_of_correct_passage >= 0:
            pred_outcome = "ID_MISCLASSIFIED_AS_OOD"
        elif (prediction >= 0 and index_of_correct_passage >= 0 and
              prediction != index_of_correct_passage):
            pred_outcome = "ID_MISCLASSIFIED_AS_ANOTHER_ID"
        else:
            raise ValueError('wrong prediction/target combination')

        out_stream.write(
            "prediction: {} / norm score {:3.3}\nprediction content:"
            "\n\t{}\n".format(
                pred_outcome,
                norm_score,
                get_passage_last_header(source2passages[source][prediction]),
            )
        )
        out_stream.write(
            "target content:\n\t{}\n\n".format(
                get_passage_last_header(source2passages[source][index_of_correct_passage])
            )
        )


def generate_embeddings(ret_trainee, input_file, out_file):
    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    source2passages, pid2passage, _ = get_passages_by_source(json_data)

    question_embs = []
    labels = []
    for example in tqdm(json_data["examples"]):
        pid = get_passage_id(example)
        passage = pid2passage[pid]
        labels.append('id' if is_in_distribution(passage) else 'ood')
        emb = ret_trainee.retriever.embed_question(get_question(example))
        question_embs.append(emb)

    passage_header_embs = []
    ood = 0
    for source, passages in source2passages.items():
        logger.info('embedding passages for source {}'.format(source))
        for passage in tqdm(passages):
            if is_in_distribution(passage):
                emb = ret_trainee.retriever.embed_paragraph(
                    get_passage_last_header(passage, return_error_for_ood=True))
                passage_header_embs.append(emb)
            else:
                ood += 1

    to_serialize = {"question_embs": question_embs, "passage_header_embs": passage_header_embs,
                    "question_labels": labels}
    with open(out_file, "wb") as out_stream:
        pickle.dump(to_serialize, out_stream)
    logger.info(
        'saved {} question embeddings and {} passage header embeddings ({} skipped because '
        'out-of-distribution)'.format(
            len(question_embs), len(passage_header_embs), ood))


def compute_result_at_threshold(
    predictions, indices_of_correct_passage, normalized_scores, threshold, log_threshold
):
    count = len(indices_of_correct_passage)
    ood_count = sum([x == -1 for x in indices_of_correct_passage])
    id_count = count - ood_count
    correct = 0
    id_correct = 0
    ood_correct = 0
    ood_misclassified_as_id = 0
    id_misclassified_as_ood = 0
    id_misclassified_as_id = 0

    for i, prediction in enumerate(predictions):
        if normalized_scores[i] >= threshold:
            after_threshold_pred = prediction
            # id_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        else:
            after_threshold_pred = -1
            # ood_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        correct += int(after_threshold_pred == indices_of_correct_passage[i])
        if indices_of_correct_passage[i] != -1 and after_threshold_pred == -1:
            # target id - prediction ood
            id_misclassified_as_ood += 1
        elif (indices_of_correct_passage[i] != -1 and
              indices_of_correct_passage[i] != after_threshold_pred):
            # target id - prediction id but wrong
            id_misclassified_as_id += 1
        elif (indices_of_correct_passage[i] != -1 and
              indices_of_correct_passage[i] == after_threshold_pred):
            # target id - prediction id and correct
            id_correct += 1
        elif indices_of_correct_passage[i] == -1 and after_threshold_pred == -1:
            # target ood - prediction ood
            ood_correct += 1
        elif indices_of_correct_passage[i] == -1 and after_threshold_pred != -1:
            # target ood - prediction id
            ood_misclassified_as_id += 1
        else:
            raise ValueError()
    acc = ((correct / count) * 100) if count > 0 else math.nan
    id_acc = ((id_correct / id_count) * 100) if id_count > 0 else math.nan
    ood_acc = ((ood_correct / ood_count) * 100) if ood_count > 0 else math.nan
    threshold_msg = "threshold {:1.3f}: ".format(threshold) if log_threshold else ""

    result_message = "\n{}overall: {:3}/{}={:3.2f}% acc".format(threshold_msg, correct, count, acc)
    result_message += "\n\tin-distribution: {:3}/{}={:3.2f}% acc".format(id_correct, id_count,
                                                                         id_acc)
    result_message += "\n\t\twrong because marked ood: {:3}/{}={:3.2f}% err".format(
        id_misclassified_as_ood, id_count,
        ((id_misclassified_as_ood / id_count) * 100) if id_count > 0 else math.nan)
    result_message += "\n\t\tmarked id but wrong candidate: {:3}/{}={:3.2f}% err".format(
        id_misclassified_as_id, id_count,
        ((id_misclassified_as_id / id_count) * 100) if id_count > 0 else math.nan)
    result_message += "\n\tout-of-distribution: {:3}/{}={:3.2f}% acc".format(
        ood_correct, ood_count, ood_acc)

    return result_message
