import hashlib
import logging
import random
from copy import deepcopy
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class Retriever(nn.Module):
    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug):
        super(Retriever, self).__init__()
        self.bert_question_encoder = bert_question_encoder
        self.bert_paragraph_encoder = bert_paragraph_encoder
        self.tokenizer = tokenizer
        self.debug = debug
        self.max_question_len = max_question_len
        self.max_paragraph_len = max_paragraph_len
        self.cache_hash2str = {}
        self.cache_hash2array = {}

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                         batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                         batch_token_type_ids_paragraphs):

        batch_size, num_document, max_len_size = batch_input_ids_paragraphs.size()

        if self.debug:
            for i in range(batch_size):
                question = self.tokenizer.convert_ids_to_tokens(
                    input_ids_question.cpu().numpy()[i])
                logger.info('>> {}'.format(question))
                for j in range(num_document):
                    answer = self.tokenizer.convert_ids_to_tokens(
                        batch_input_ids_paragraphs.cpu().numpy()[i][j])
                    logger.info('>>>> {}'.format(answer))

        h_question = self.bert_question_encoder(
            input_ids=input_ids_question, attention_mask=attention_mask_question,
            token_type_ids=token_type_ids_question)

        batch_input_ids_paragraphs_reshape = batch_input_ids_paragraphs.reshape(
            -1, max_len_size)
        batch_attention_mask_paragraphs_reshape = batch_attention_mask_paragraphs.reshape(
            -1, max_len_size)
        batch_token_type_ids_paragraphs_reshape = batch_token_type_ids_paragraphs.reshape(
            -1, max_len_size)

        h_paragraphs_batch_reshape = self.bert_paragraph_encoder(
            input_ids=batch_input_ids_paragraphs_reshape,
            attention_mask=batch_attention_mask_paragraphs_reshape,
            token_type_ids=batch_token_type_ids_paragraphs_reshape)
        h_paragraphs_batch = h_paragraphs_batch_reshape.reshape(batch_size, num_document, -1)
        return h_question, h_paragraphs_batch

    def str2hash(drlf, str):
        return hashlib.sha224(str.encode('utf-8')).hexdigest()

    def refresh_cache(self):
        self.cache_hash2array = {}
        self.cache_hash2str = {}

    def predict(self, question_str: str, batch_paragraph_strs: List[str], refresh_cache = False):
        self.eval()
        with torch.no_grad():
            ## TODO this is only a single batch
            ## TODO Add hashing

            paragraph_inputs = self.tokenizer.batch_encode_plus(
                 batch_paragraph_strs,
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 max_length=self.max_paragraph_len,
                 return_tensors='pt'
             )

            tmp_device = next(self.bert_paragraph_encoder.parameters()).device
            p_inputs = {k: v.to(tmp_device).unsqueeze(0) for k,v in paragraph_inputs.items()}

            question_inputs = self.tokenizer.encode_plus(question_str, add_special_tokens=True,
                   max_length=self.max_question_len, pad_to_max_length=True,
                   return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            q_inputs = {k: v.to(tmp_device) for k,v in question_inputs.items()}

            q_emb, p_embs = self.forward(
                q_inputs['input_ids'], q_inputs['attention_mask'], q_inputs['token_type_ids'],
                p_inputs['input_ids'], p_inputs['attention_mask'], p_inputs['token_type_ids'],
            )

            relevance_scores = torch.sigmoid(
                torch.matmul(q_emb, p_embs.squeeze(0).T).squeeze(0)
            )

            rerank_index = torch.argsort(-relevance_scores)
            relevance_scores_numpy = relevance_scores.detach().cpu().numpy()
            rerank_index_numpy = rerank_index.detach().cpu().numpy()
            reranked_paragraphs = [batch_paragraph_strs[i] for i in rerank_index_numpy]
            reranked_relevance_scores = relevance_scores_numpy[rerank_index_numpy]
            return reranked_paragraphs, reranked_relevance_scores, rerank_index_numpy


class RetrieverTrainer(pl.LightningModule):

    def __init__(self, retriever, train_data, dev_data, test_data, loss_type,
                 optimizer_type):
        super(RetrieverTrainer, self).__init__()
        self.retriever = retriever
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, **kwargs):
        return self.retriever(**kwargs)

    def step_helper(self, batch):
        input_ids_question, attention_mask_question, token_type_ids_question, \
        batch_input_ids_paragraphs, batch_attention_mask_paragraphs, \
        batch_token_type_ids_paragraphs, targets = batch

        inputs = {
            'input_ids_question': input_ids_question,
            'attention_mask_question': attention_mask_question,
            'token_type_ids_question': token_type_ids_question,
            'batch_input_ids_paragraphs': batch_input_ids_paragraphs,
            'batch_attention_mask_paragraphs': batch_attention_mask_paragraphs,
            'batch_token_type_ids_paragraphs': batch_token_type_ids_paragraphs
        }

        q_emb, p_embs = self(**inputs)
        batch_size, num_document, emb_dim = p_embs.size()
        all_dots = torch.bmm(q_emb.unsqueeze(1), p_embs.transpose(2, 1)).squeeze(1)
        all_prob = torch.sigmoid(all_dots)

        if self.loss_type == 'negative_sampling':
            raise ValueError('need to fix now that we have target that is not always 0')
            pos_loss = - torch.log(all_prob[:, 0]).sum()
            neg_loss = - torch.log(1 - all_prob[:, 1:]).sum()
            loss = pos_loss + neg_loss
        elif self.loss_type == 'classification':
            # all_dots are the logits
            loss = self.cross_entropy(all_dots, targets)
        elif self.loss_type == 'triplet_loss':
            raise ValueError('need to fix now that we have target that is not always 0')
            assert p_embs.shape[1] == 3
            # picking a random negative example
            negative_index = random.randint(1, 2)
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            loss = triplet_loss(q_emb,
                                p_embs[:, 0, :],
                                p_embs[:, negative_index, :])
        elif self.loss_type == 'cosine':
            raise ValueError('need to fix now that we have target that is not always 0')
            targets = torch.ones(batch_size, num_document)
            # first target stays as 1 (we want those vectors to be similar)
            # other targets -1 (we want them to be far away)
            targets[:, 1:] *= -1
            targets = targets.reshape(-1).to(q_emb.device)
            q_emb.repeat(num_document, 1), p_embs.reshape(-1, emb_dim)
            loss = torch.nn.CosineEmbeddingLoss()(
                q_emb.repeat(num_document, 1), p_embs.reshape(-1, emb_dim),
                targets
            )
        else:
            raise ValueError('loss_type {} not supported. Please choose between negative_sampling,'
                             ' classification, cosine')
        return loss, all_prob

    def training_step(self, batch, batch_idx):
        """
        batch comes in the order of question, 1 positive paragraph,
        K negative paragraphs
        """

        train_loss, _ = self.step_helper(batch)
        # logs
        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def training_step_end(self, outputs):
        loss_value = outputs['loss'].mean()
        tensorboard_logs = {'train_loss': loss_value}
        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, all_prob = self.step_helper(batch)
        batch_size = all_prob.size()[0]
        _, predictions = torch.max(all_prob, 1)
        targets = batch[-1]
        val_acc = torch.tensor(accuracy_score(targets.cpu(), predictions.cpu())).to(targets.device)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).double().mean()

        tqdm_dict = {'val_acc': avg_val_acc, 'val_loss': avg_val_loss}

        results = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        return results

    def test_step(self, batch, batch_idx):
        # we do the same stuff as in the validation phase
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.optimizer_type == 'adamw':
            return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad])
        elif self.optimizer_type == 'adam':
            return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=0.0001)
        else:
            raise ValueError('optimizer {} not supported'.format(self.optimizer_type))

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.dev_data

    def test_dataloader(self):
        return self.test_data
