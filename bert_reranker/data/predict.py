import json
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_batched_pairs(qa_pairs, batch_size):
    result = []
    for i in range(0, len(qa_pairs), batch_size):
        result.append(qa_pairs[i:i + batch_size])
    return result


def evaluate_model(ret_trainee, qa_pairs_json_file, predict_to, ground_truth_available):

    with open(qa_pairs_json_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    correct = 0
    count = 0
    out_stream = open(predict_to, 'w') if predict_to else None

    if not ground_truth_available:
        out_stream.write('!!NO GROUND TRUTH AVAILABLE FOR THESE PREDICTIONS!!\n\n')
    for question, answers in tqdm(qa_pairs):

        out = ret_trainee.retriever.predict(question, answers)

        if out[2][0] == 0:  # answers[0] is always the correct answer
            correct += 1

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
        count += 1

    if ground_truth_available:
        acc = correct / len(qa_pairs) * 100
        logger.info("correct {} over {} - accuracy is {}".format(correct, len(qa_pairs), acc))
    else:
        logger.info("--ground-truth-available not used - not computing accuracy")
    if out_stream:
        out_stream.close()
