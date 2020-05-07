import json
import logging
import math

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_batched_pairs(qa_pairs, batch_size):
    result = []
    for i in range(0, len(qa_pairs), batch_size):
        result.append(qa_pairs[i:i + batch_size])
    return result


def generate_predictions(ret_trainee, qa_pairs_json_file, predict_to, ground_truth_available):

    with open(qa_pairs_json_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    predictions = []
    normalized_scores = []
    out_stream = open(predict_to, 'w') if predict_to else None

    if not ground_truth_available and out_stream is not None:
        out_stream.write('!!NO GROUND TRUTH AVAILABLE FOR THESE PREDICTIONS!!\n\n')
    for question, answers in tqdm(qa_pairs):

        out = ret_trainee.retriever.predict(question, answers)

        predictions.append(out[2][0])
        normalized_scores.append(out[3][0])

        if out_stream:
            out_stream.write('-------------------------\n')
            out_stream.write('question:\n\t{}\n'.format(question))
            if ground_truth_available:
                out_stream.write(
                    'prediction: correct? {} / score {:3.3} / norm score {:3.3} / answer content:'
                    '\n\t{}\n'.format(
                        out[2][0] == 0, out[1][0], out[3][0], out[0][0]))
                out_stream.write(
                    'ground truth:\n\t{}\n\n'.format(answers[0]))
            else:
                out_stream.write(
                    'prediction: score {:3.3} / norm score {:3.3} / answer content:'
                    '\n\t{}\n'.format(out[1][0], out[3][0], out[0][0]))

    if ground_truth_available:
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            result_message = compute_result_at_threshold(predictions, normalized_scores, threshold)
            logger.info(result_message)
            if out_stream is not None:
                out_stream.write(result_message + '\n')
    else:
        logger.info("--ground-truth-available not used - not computing accuracy")
    if out_stream:
        out_stream.close()


def compute_result_at_threshold(predictions, normalized_scores, threshold):
    correct = 0
    count = 0
    not_considered = 0
    for i, prediction in enumerate(predictions):
        if normalized_scores[i] >= threshold:
            correct += int(prediction == 0)
            count += 1
        else:
            not_considered += 1
    acc = correct / count * 100 if count > 0 else math.nan
    return "threshold {:1.3f}: entries included: {:4} (filtered out :{:4}) - correct " \
           "(among the included): {:4} - accuracy is {:3.2f}".format(
               threshold, count, not_considered, correct, acc)
