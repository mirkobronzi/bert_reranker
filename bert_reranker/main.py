#!/usr/bin/env python

import argparse
import logging
import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from yaml import load

from bert_reranker.data.data_loader import generate_dataloader
from bert_reranker.data.predict import evaluate_model
from bert_reranker.models.load_model import load_model
from bert_reranker.models.pl_model_loader import try_to_restore_model_weights
from bert_reranker.models.retriever import RetrieverTrainer
from bert_reranker.utils.hp_utils import check_and_log_hp
from bert_reranker.utils.logging_utils import LoggerWriter

logger = logging.getLogger(__name__)
from pytorch_lightning import loggers

import numpy as np

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
    parser.add_argument('--train', help='will not train - will just evaluate on dev',
                        action='store_true')
    parser.add_argument('--validate', help='will not train - will just evaluate on dev',
                        action='store_true')
    parser.add_argument('--predict', help='will predict on the json file you provide as an arg')
    parser.add_argument('--predict-to', help='(optiona) write predictions here)')
    parser.add_argument('--redirect-log', help='will intercept any stdout/err and log it',
                        action='store_true')
    parser.add_argument('--debug', help='will log more info', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    
    if args.redirect_log:
        sys.stdout = LoggerWriter(logger.info)
        sys.stderr = LoggerWriter(logger.warning)

    with open(args.config, 'r') as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)

    check_and_log_hp(
        ['train_file', 'dev_files', 'test_file', 'cache_folder', 'batch_size', 'tokenizer_name', 'model',
         'max_question_len', 'max_paragraph_len', 'patience', 'gradient_clipping',
         'loss_type', 'optimizer',  'precision', 'accumulate_grad_batches', 'seed'],
        hyper_params)
    
    if hyper_params['seed'] is not None:
        # fix the seed
        torch.manual_seed(hyper_params['seed'])
        np.random.seed(hyper_params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(hyper_params['cache_folder'], exist_ok=True)

    tokenizer_name = hyper_params['tokenizer_name']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataloader = generate_dataloader(
        hyper_params['train_file'], hyper_params['cache_folder'],
        hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
        tokenizer, hyper_params['batch_size'])

    dev_dataloaders = []
    for dev_file in hyper_params['dev_files'].values():
        dev_dataloaders.append(
            generate_dataloader(
                dev_file,
                hyper_params['cache_folder'],
                hyper_params['max_question_len'],
                hyper_params['max_paragraph_len'],
                tokenizer, hyper_params['batch_size']
            )
        )

    test_dataloader = generate_dataloader(
        hyper_params['test_file'], hyper_params['cache_folder'],
        hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
        tokenizer, hyper_params['batch_size'])

    ret = load_model(hyper_params, tokenizer, args.debug)

    os.makedirs(args.output, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.output, '{epoch}-{val_acc_0:.2f}-{val_loss_0:.2f}'),
        save_top_k=1,
        verbose=True,
        monitor='val_acc_0',
        mode='max'
    )

    early_stopping = EarlyStopping('val_acc_0', mode='max', patience=hyper_params['patience'])

    if hyper_params['precision'] not in {16, 32}:
        raise ValueError('precision should be either 16 or 32')

    if not args.no_model_restoring:
        ckpt_to_resume = try_to_restore_model_weights(args.output)
    else:
        ckpt_to_resume = None
        logger.info('will not try to restore previous models because --no-model-restoring')

    tb_logger = loggers.TensorBoardLogger('logs')
    for hparam in list(hyper_params):
        tb_logger.experiment.add_text(hparam, str(hyper_params[hparam]))

    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=args.gpu,
        distributed_backend='dp',
        val_check_interval=args.validation_interval,
        min_epochs=1,
        gradient_clip_val=hyper_params['gradient_clipping'],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        precision=hyper_params['precision'],
        resume_from_checkpoint=ckpt_to_resume,
        accumulate_grad_batches=hyper_params['accumulate_grad_batches'])

    ret_trainee = RetrieverTrainer(ret, train_dataloader, dev_dataloaders, test_dataloader,
                                   hyper_params['loss_type'], hyper_params['optimizer'])

    if args.train:
        trainer.fit(ret_trainee)
    elif args.validate:
        trainer.test(ret_trainee)
    elif args.predict:
        model_ckpt = torch.load(
            ckpt_to_resume, map_location=torch.device("cpu")
        )
        ret_trainee.load_state_dict(model_ckpt["state_dict"])
        evaluate_model(ret_trainee, qa_pairs_json_file=args.predict, predict_to=args.predict_to)
    else:
        logger.warning('please select one between --train / --validate / --test')


if __name__ == '__main__':
    main()
