import json
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_batched_pairs(qa_pairs, batch_size):
    result = []
    for i in range(0, len(qa_pairs), batch_size):
        result.append(qa_pairs[i:i + batch_size])
    return result


def evaluate_model(ret_trainee, qa_pairs_json_file, predict_to):

    with open(qa_pairs_json_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    correct = 0
    out_stream = open(predict_to, 'w') if predict_to else None
    for question, answers in tqdm(qa_pairs):

        out = ret_trainee.retriever.predict(question, answers)

        if out[2][0] == 0:  # answers[0] is always the correct answer
            correct += 1

        if out_stream:
            out_stream.write(question + '\n')
            out_stream.write('ranking: {}'.format(out[2]) + '\n')
            out_stream.write('\t' + '\n\t'.join(answers) + '\n\n')

    acc = correct / len(qa_pairs) * 100
    logger.info("correct {} over {} - accuracy is {}".format(correct, len(qa_pairs), acc))
    if out_stream:
        out_stream.close()
