#!/usr/bin/env python

import argparse
import logging
import json
import re
import random
from tqdm import tqdm
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from yaml import load

from bert_reranker.data.data_loader import generate_natq_dataloaders
from bert_reranker.models.bert_encoder import BertEncoder
from bert_reranker.models.pl_model_loader import try_to_restore_model_weights
from bert_reranker.models.retriever import Retriever, RetrieverTrainer
from bert_reranker.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)

def remove_html_tags(data):
    p = re.compile(r"<.*?>")
    return p.sub("", data)


def make_qa_pairs_natq(natq_path, n_samples=100):
    with open(natq_path) as json_file:
        contents = json_file.readlines()
        natq = [json.loads(cont) for cont in contents]

    qa_pairs = []
    for ii in range(n_samples):
        question = remove_html_tags(natq[ii]["question"])
        correct_answer = remove_html_tags(natq[ii]["right_paragraphs"][0])
        try:
            wrong_answers = [
                remove_html_tags(natq[ii]["wrong_paragraphs"][0]),
                remove_html_tags(natq[ii]["wrong_paragraphs"][1]),
            ]
        except IndexError:
            # For some reason it can happen that there are not enough
            # wrong paragraphs, so we assume it's insanely unlikely the
            # right paragraph from the previous case would be right again.
            wrong_answers = [
                remove_html_tags(natq[ii]["wrong_paragraphs"][0]),
                remove_html_tags(natq[ii - 1]["right_paragraphs"][0]),
            ]

        candidate_answers = []
        candidate_answers.append(correct_answer)
        candidate_answers.extend(wrong_answers)

        qa_pairs.append([question, candidate_answers])

    return qa_pairs


def make_qa_pairs_faq(faq_path, n_wrong_answers=2, seed=42):
    with open(faq_path, "r") as fh:
        faq = json.load(fh)

    random.seed(seed)
    all_questions = []
    all_answers = []

    for k, v in faq.items():
        if k != "document_URL":
            all_questions.append(k)
            all_answers.append("".join(faq[k]["plaintext"]))

    qa_pairs = []
    for idx, question in enumerate(all_questions):
        correct_answer = all_answers[idx]
        wrong_answers = all_answers.copy()
        wrong_answers.remove(correct_answer)
        random.shuffle(wrong_answers)

        candidate_answers = []
        candidate_answers.append(correct_answer)
        candidate_answers.extend(wrong_answers[:n_wrong_answers])
        qa_pairs.append([question, candidate_answers])

    return qa_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='config file with generic hyper-parameters,  such as optimizer, '
                             'batch_size, ... -  in yaml format', required=True)
    parser.add_argument('--gpu', help='list of gpu ids to use. default is cpu. example: --gpu 0 1',
                        type=int, nargs='+', default=0)
    parser.add_argument('--validation-interval', help='how often to run validation in one epoch - '
                                                      'e.g., 0.5 means halfway - default 0.5',
                        type=float, default=0.5)
    parser.add_argument('--output', help='where to store models', required=True)
    parser.add_argument('--no-model-restoring', help='will not restore any previous model weights ('
                                                     'even if present)', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config, 'r') as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)

    check_and_log_hp(
        ['natq_json_file', 'cache_folder', 'batch_size', 'model_name', 'max_question_len',
         'max_paragraph_len', 'embedding_dim', 'patience', 'gradient_clipping', 'loss_type',
         'optimizer_type', 'freeze_bert', 'pooling_type'],
        hyper_params)

    os.makedirs(hyper_params['cache_folder'], exist_ok=True)

    model_name = hyper_params['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataloader, dev_dataloader = generate_natq_dataloaders(
        hyper_params['natq_json_file'], hyper_params['cache_folder'],
        hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
        tokenizer, hyper_params['batch_size'])

    bert_question = AutoModel.from_pretrained(model_name)
    bert_paragraph = AutoModel.from_pretrained(model_name)

    bert_question_encoder = BertEncoder(bert_question, hyper_params['max_question_len'],
                                        hyper_params['embedding_dim'], hyper_params['freeze_bert'],
                                        hyper_params['pooling_type'])
    bert_paragraph_encoder = BertEncoder(bert_paragraph, hyper_params['max_paragraph_len'],
                                         hyper_params['embedding_dim'], hyper_params['freeze_bert'],
                                         hyper_params['pooling_type'])

    ret = Retriever(bert_question_encoder, bert_paragraph_encoder, tokenizer,
                    hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
                    hyper_params['embedding_dim'])
    os.makedirs(args.output, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.output, '{epoch}-{val_loss:.2f}-{val_acc:.2f}'),
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    early_stopping = EarlyStopping('val_acc', mode='max', patience=hyper_params['patience'])

    if not args.no_model_restoring:
        ckpt_to_resume = try_to_restore_model_weights(args.output)

    else:
        ckpt_to_resume = None
        logger.info('will not try to restore previous models because --no-model-restoring')

    trainer = pl.Trainer(
        gpus=args.gpu,
        distributed_backend='dp',
        val_check_interval=args.validation_interval,
        min_epochs=1,
        max_epochs=2,
        gradient_clip_val=hyper_params['gradient_clipping'],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        resume_from_checkpoint=ckpt_to_resume)

    ret_trainee = RetrieverTrainer(ret, train_dataloader, dev_dataloader,
                                   hyper_params['embedding_dim'],
                                   hyper_params['loss_type'],
                                   hyper_params['optimizer_type'])

    ret.predict("Hey", ["sup", "nm you?", "nm bro thanks"], refresh_cache=True)
    #  trainer.fit(ret_trainee)
    #  trainer.save_checkpoint('random_checkpoint.pth')

    faq_path = "/home/jerpint/covidfaq/covidfaq/scrape/quebec-en-faq.json"
    natq_path = "/home/jerpint/covidfaq/covidfaq/data/natq_clean.json"
    n_wrong_answers = 2  # number of wrong answers added to the correct answer

    # Run the test on samples from natq to sanity check evreything is correct
    qa_pairs_natq = make_qa_pairs_natq(natq_path, n_samples=50)
    correct = 0
    for question, answers in tqdm(qa_pairs_natq):

        out = ret.predict(question, answers)

        if out[2][0] == 0:  # answers[0] is always the correct answer
            correct += 1

    acc = correct / len(qa_pairs_natq) * 100
    print("single run accuracy natq: %", acc)

    # Run the test on 10 separate splits of the FAQ and average the results
    accs = []
    for seed in range(10):
        qa_pairs_faq = make_qa_pairs_faq(
            faq_path, n_wrong_answers=n_wrong_answers, seed=seed
        )
        correct = 0
        for question, answers in tqdm(qa_pairs_faq):

            out = ret.predict(question, answers)
            #  out = model.retriever.predict(question.lower(), [answer.lower() for answer in answers])

            if out[2][0] == 0:  # answers[0] is always the correct answer
                correct += 1

        acc = correct / len(qa_pairs_faq) * 100
        accs.append(acc)
        print("single run accuracy: %", acc)

    print("Average model accuracy on FAQ: ", sum(accs) / len(accs))

if __name__ == '__main__':
    main()
