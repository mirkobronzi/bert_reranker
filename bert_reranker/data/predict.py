import json
import logging
import math
import pickle

import pandas as pd
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


def generate_embeddings(ret_trainee, input_file, out_file):
    if input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
        if 'samasource' in input_file:
            not_user_health_related = (1- (data.samasource_annotation == 'USER_HEALTH_RELATED'))
            not_user_health_related = not_user_health_related.astype('bool')
            data = data.loc[not_user_health_related, :]
        
            q_embs = []
            questions = []
            clusters = []
            samasource_annotations = []
            for question, cluster, sam_ant in tqdm(
                    zip(data.question_processed, data.cluster, data.samasource_annotation)):
                q_emb = ret_trainee.retriever.embed_question(question)
                q_embs.append(q_emb)
                questions.append(question)
                clusters.append(cluster)
                samasource_annotations.append(sam_ant)

            dct = {'embs': q_embs, 'questions': questions, 
                   'clusters': clusters, 'annotations' : samasource_annotations}
            with open(out_file, 'wb') as out_stream:
                pickle.dump(dct, out_stream)

        else:
            
            q_embs = []
            vals = []
            topics = []
            gt_questions = []
            for question, validation, topic, gt_question in tqdm(
                    zip(data.question, data.validation, data.topic, data.gt_question), total=len(data)):
                q_emb = ret_trainee.retriever.embed_question(question)
                q_embs.append(q_emb)
                vals.append(validation)
                topics.append(topic)
                gt_questions.append(gt_question)

            dct = {'embs': q_embs, 'vals': vals, 
                   'topics': topics, 'gt_questions' : gt_questions}
            with open('QUESTION_' + out_file, 'wb') as out_stream:
                pickle.dump(dct, out_stream)

            logger.info('embedded {} questions - now embedding {} gt_questions'.format(
                        len(q_embs), len(gt_questions)))
            p_embs = []
            
            for gt_question in tqdm(set(gt_questions)):
                p_emb = ret_trainee.retriever.embed_paragraph(gt_question)
                p_embs.append(p_emb)
            dct_gt = {'embs' : p_embs, 
                      'gt_questions' : set(gt_questions)}
            with open('GT_QUESTION_' + out_file, 'wb') as out_stream:
                pickle.dump(dct_gt, out_stream)


    elif input_file.endswith('json'):
        import pdb
        data = pd.read_json(input_file)
        
        gt_questions = list(data.iloc[:, 0].values)

        p_embs = []
        for gt_question in tqdm(gt_questions):
            p_emb = ret_trainee.retriever.embed_paragraph(gt_question)
            p_embs.append(p_emb)

        dct_gt = {'embs' : p_embs, 
                  'gt_questions' : (gt_questions)}
        with open(out_file, 'wb') as out_stream:
            pickle.dump(dct_gt, out_stream)

    else:
        raise ValueError('I do not support that extension, go somewhere else')


def compute_result_at_threshold(predictions, indices_of_correct_passage, normalized_scores,
                                threshold):
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
    return "threshold {:1.3f}: overall correct: {:3}/{:3}={:3.2f} - in-distribution correct" \
           "{:3}/{:3}={:3.2f} - out-of-distribution: {:3}/{:3}={:3.2f}".format(
               threshold, correct, count, acc, id_correct, id_count, id_acc, ood_correct, ood_count,
               ood_acc)
