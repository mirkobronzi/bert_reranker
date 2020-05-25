import json
import logging
import math

from tqdm import tqdm

from bert_reranker.data.data_loader import get_passages_by_source, _encode_passages, \
    get_passage_text

logger = logging.getLogger(__name__)


def get_batched_pairs(qa_pairs, batch_size):
    result = []
    for i in range(0, len(qa_pairs), batch_size):
        result.append(qa_pairs[i:i + batch_size])
    return result


def generate_predictions(ret_trainee, json_file, predict_to):

    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    predictions = []
    normalized_scores = []
    indices_of_correct_passage = []
    out_stream = open(predict_to, 'w') if predict_to else None

    source2passages, passage_id2source, passage_id2index = get_passages_by_source(json_data)
    source2encoded_passages, _, _ = _encode_passages(
        source2passages, ret_trainee.retriever.max_question_len, ret_trainee.retriever.tokenizer)
    for example in tqdm(json_data['examples']):
        question = example['question']
        source = example['source']
        index_of_correct_passage = passage_id2index[example['passage_id']]

        prediction, norm_score = ret_trainee.retriever.predict(
            question, source2encoded_passages[source])

        predictions.append(prediction)
        normalized_scores.append(norm_score)
        indices_of_correct_passage.append(index_of_correct_passage)

        if out_stream:
            out_stream.write('-------------------------\n')
            out_stream.write('question:\n\t{}\n'.format(question))
            out_stream.write(
                'prediction: correct? {} / norm score {:3.3} / answer content:'
                '\n\t{}\n'.format(
                    prediction == index_of_correct_passage, norm_score,
                    get_passage_text(source2passages[source][prediction])))
            out_stream.write(
                'ground truth:\n\t{}\n\n'.format(
                    get_passage_text(source2passages[source][index_of_correct_passage])))

    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        result_message = compute_result_at_threshold(predictions, indices_of_correct_passage,
                                                     normalized_scores, threshold)
        logger.info(result_message)
        if out_stream is not None:
            out_stream.write(result_message + '\n')

    if out_stream:
        out_stream.close()


def compute_result_at_threshold(predictions, indices_of_correct_passage, normalized_scores,
                                threshold):
    correct = 0
    count = 0
    not_considered = 0
    for i, prediction in enumerate(predictions):
        if normalized_scores[i] >= threshold:
            correct += int(prediction == indices_of_correct_passage[i])
            count += 1
        else:
            not_considered += 1
    acc = correct / count * 100 if count > 0 else math.nan
    return "threshold {:1.3f}: entries included: {:4} (filtered out :{:4}) - correct " \
           "(among the included): {:4} - accuracy is {:3.2f}".format(
               threshold, count, not_considered, correct, acc)
