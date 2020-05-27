#!/usr/bin/env python

import argparse
import logging
import os
import pickle
import sys
from logging.handlers import WatchedFileHandler

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from orion.client import report_results
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from yaml import load

from bert_reranker.data.data_loader import generate_dataloader
from bert_reranker.data.predict import generate_embeddings, Predictor, PredictorWithOutlierDetector
from bert_reranker.models.cache_manager import CacheManagerCallback
from bert_reranker.models.load_model import load_model
from bert_reranker.models.pl_model_loader import try_to_restore_model_weights
from bert_reranker.models.retriever_trainer import RetrieverTrainer
from bert_reranker.models.sklearn_outliers_model import SKLEARN_MODEL_FILE_NAME
from bert_reranker.utils.hp_utils import check_and_log_hp
from bert_reranker.utils.logging_utils import LoggerWriter
from bert_reranker.scripts.tokenizer_cutoff import evaluate_tokenizer_cutoff

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="config file with generic hyper-parameters,  such as optimizer, "
        "batch_size, ... -  in yaml format",
        required=True,
    )
    parser.add_argument(
        "--gpu",
        help="list of gpu ids to use. default is cpu. example: --gpu 0 1",
        type=int,
        nargs="+"
    )
    parser.add_argument(
        "--validation-interval",
        help="how often to run validation in one epoch - "
        "e.g., 0.5 means halfway - default 0.5",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output", help="where to store models", required=True)
    parser.add_argument(
        "--no-model-restoring",
        help="will not restore any previous model weights (" "even if present)",
        action="store_true",
    )
    parser.add_argument(
        "--train",
        help="will not train - will just evaluate on dev",
        action="store_true",
    )
    parser.add_argument(
        "--validate",
        help="will not train - will just evaluate on dev",
        action="store_true",
    )
    parser.add_argument(
        "--predict", help="will predict on the json file you provide as an arg"
    )
    parser.add_argument(
        "--predict-outliers", help="will use the sklearn model to predict outliers",
        action="store_true"
    )
    parser.add_argument(
        "--file-to-emb", help="will use this file as input to generate embeddings"
    )
    parser.add_argument(
        "--write-emb-to", help="will write question embeddings to this file"
    )
    parser.add_argument(
        "--save-weights-to",
        help="will save ONLY the model weights (not the pytorch lightning object)"
        " to this file",
    )
    parser.add_argument("--predict-to", help="(optional) write predictions here)")
    parser.add_argument(
        "--redirect-log",
        help="will intercept any stdout/err and log it",
        action="store_true",
    )
    parser.add_argument(
        "--num-workers", help="number of workers - default 2", type=int, default=0
    )
    parser.add_argument(
        "--print-sentence-stats",
        help="will print stats on the data",
        action="store_true",
    )
    parser.add_argument(
        "--multiple-thresholds",
        help="will print results for various thresholds",
        action="store_true",
    )
    parser.add_argument('--log', help='log to this file (in addition to stdout/err)')
    parser.add_argument("--debug", help="will log more info", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # will log to a file if provided (useful for orion on cluster)
    if args.log is not None:
        handler = WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
    if args.redirect_log or args.log:
        sys.stdout = LoggerWriter(logger.info)
        sys.stderr = LoggerWriter(logger.warning)

    with open(args.config, "r") as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)

    if args.gpu is None and 'GPU' in os.environ:
        gpu_string = os.environ['GPU']
        gpu = [int(x) for x in gpu_string.strip().split()]
    else:
        gpu = args.gpu

    ckpt_to_resume, ret_trainee, trainer = init_model(
        hyper_params,
        args.num_workers,
        args.output,
        args.validation_interval,
        gpu,
        args.no_model_restoring,
        args.debug,
        args.print_sentence_stats
    )

    if args.train:
        trainer.fit(ret_trainee)
        best_dev_result = float(trainer.early_stop_callback.best.cpu().numpy())
        report_results([dict(
            name='dev_metric',
            type='objective',
            # note the minus - cause orion is always trying to minimize (cit. from the guide)
            value=-float(best_dev_result))])
    elif args.validate:
        trainer.test(ret_trainee)
    elif args.predict:
        if not args.predict_to:
            raise ValueError('--predict also requires --predict-to')

        model_ckpt = torch.load(ckpt_to_resume, map_location=torch.device("cpu"))
        ret_trainee.load_state_dict(model_ckpt["state_dict"])
        if args.predict_outliers:
            with open(os.path.join(args.output, SKLEARN_MODEL_FILE_NAME), 'rb') as file:
                sklearn_model = pickle.load(file)
            predictor = PredictorWithOutlierDetector(ret_trainee, sklearn_model)
        else:
            predictor = Predictor(ret_trainee)
        predictor.generate_predictions(
            json_file=args.predict,
            predict_to=args.predict_to,
            multiple_thresholds=args.multiple_thresholds
        )
    elif args.file_to_emb:
        if args.write_emb_to is None:
            raise ValueError('please specify also --write-emb-to')
        model_ckpt = torch.load(ckpt_to_resume, map_location=torch.device("cpu"))
        ret_trainee.load_state_dict(model_ckpt["state_dict"])
        generate_embeddings(
            ret_trainee,
            input_file=args.file_to_emb,
            out_file=args.write_emb_to
        )
    elif args.save_weights_to is not None:
        torch.save(ret_trainee.retriever.state_dict(), args.save_weights_to)
    else:
        logger.warning("please select one between --train / --validate / --test")


def init_model(
    hyper_params,
    num_workers,
    output,
    validation_interval,
    gpu,
    no_model_restoring,
    debug,
    print_sentence_stats
):

    check_and_log_hp(
        [
            "train_file",
            "dev_files",
            "test_file",
            "batch_size",
            "tokenizer_name",
            "model",
            "max_question_len",
            "max_paragraph_len",
            "patience",
            "gradient_clipping",
            "max_epochs",
            "loss_type",
            "optimizer",
            "precision",
            "accumulate_grad_batches",
            "seed",
            "logging",
        ],
        hyper_params,
    )

    if hyper_params["seed"] is not None:
        # fix the seed
        torch.manual_seed(hyper_params["seed"])
        np.random.seed(hyper_params["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer_name = hyper_params["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ret = load_model(hyper_params, tokenizer, debug)

    os.makedirs(output, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output, "{epoch}-{val_acc_0:.2f}-{val_loss_0:.2f}"),
        save_top_k=1,
        verbose=True,
        monitor="val_acc_0",
        mode="max",
        period=0,
    )
    early_stopping = EarlyStopping(
        "val_acc_0", mode="max", patience=hyper_params["patience"]
    )

    if (
        hyper_params["model"].get("name") == "bert_encoder"
        and hyper_params["model"].get("cache_size", 0) > 0
    ):
        cbs = [CacheManagerCallback(ret, output)]
    else:
        cbs = []

    if hyper_params["precision"] not in {16, 32}:
        raise ValueError("precision should be either 16 or 32")
    if not no_model_restoring:
        ckpt_to_resume = try_to_restore_model_weights(output)
    else:
        ckpt_to_resume = None
        logger.info(
            "will not try to restore previous models because --no-model-restoring"
        )
    if hyper_params["logging"]["logger"] == "tensorboard":
        pl_logger = loggers.TensorBoardLogger("experiment_logs")
        for hparam in list(hyper_params):
            pl_logger.experiment.add_text(hparam, str(hyper_params[hparam]))
    elif hyper_params["logging"]["logger"] == "wandb":
        orion_trial_id = os.environ.get('ORION_TRIAL_ID')
        name = orion_trial_id if orion_trial_id else hyper_params["logging"]["name"]
        pl_logger = WandbLogger(
            name=name,
            project=hyper_params["logging"]["project"],
            group=hyper_params["logging"]["group"],
        )
        pl_logger.log_hyperparams(hyper_params)
    else:
        raise ValueError(
            logger.info(
                "logger {} is not implemnted".format(hyper_params["logging"]["logger"])
            )
        )

    trainer = pl.Trainer(
        logger=pl_logger,
        gpus=gpu,
        distributed_backend="dp",
        val_check_interval=validation_interval,
        min_epochs=1,
        gradient_clip_val=hyper_params["gradient_clipping"],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        callbacks=cbs,
        precision=hyper_params["precision"],
        resume_from_checkpoint=ckpt_to_resume,
        accumulate_grad_batches=hyper_params["accumulate_grad_batches"],
        max_epochs=hyper_params["max_epochs"],
    )

    dev_dataloaders, test_dataloader, train_dataloader = get_data_loaders(
        hyper_params, num_workers, tokenizer
    )

    if print_sentence_stats:
        evaluate_tokenizer_cutoff(
            hyper_params["train_file"],
            tokenizer,
            hyper_params["max_question_len"],
            hyper_params["max_paragraph_len"],
        )

    ret_trainee = RetrieverTrainer(
        ret,
        train_dataloader,
        dev_dataloaders,
        test_dataloader,
        hyper_params["loss_type"],
        hyper_params["optimizer"],
    )
    return ckpt_to_resume, ret_trainee, trainer


def get_data_loaders(hyper_params, num_workers, tokenizer):
    train_dataloader = generate_dataloader(
        hyper_params["train_file"],
        hyper_params["max_question_len"],
        hyper_params["max_paragraph_len"],
        tokenizer,
        hyper_params["batch_size"],
        num_workers=num_workers,
        shuffle=True,
    )
    dev_dataloaders = []
    for dev_file in hyper_params["dev_files"].values():
        dev_dataloaders.append(
            generate_dataloader(
                dev_file,
                hyper_params["max_question_len"],
                hyper_params["max_paragraph_len"],
                tokenizer,
                hyper_params["batch_size"],
                num_workers=num_workers,
                shuffle=False,
            )
        )
    test_dataloader = generate_dataloader(
        hyper_params["test_file"],
        hyper_params["max_question_len"],
        hyper_params["max_paragraph_len"],
        tokenizer,
        hyper_params["batch_size"],
        num_workers=num_workers,
        shuffle=False,
    )
    return dev_dataloaders, test_dataloader, train_dataloader


if __name__ == "__main__":
    main()
