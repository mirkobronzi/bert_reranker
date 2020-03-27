import hashlib
from copy import deepcopy
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class Retriever(nn.Module):
    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, emb_dim):
        super(Retriever, self).__init__()
        self.bert_question_encoder = bert_question_encoder
        self.bert_paragraph_encoder = bert_paragraph_encoder
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_paragraph_len = max_paragraph_len
        self.emb_dim = emb_dim
        self.cache_hash2str = {}
        self.cache_hash2array = {}

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                         batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                         batch_token_type_ids_paragraphs):

        batch_size, num_document, max_len_size = batch_input_ids_paragraphs.size()

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
            ## Todo: embed all unique docs, then create ranking for all questions, then find overlap with constrained ranking
            batch_paragraph_array = np.random.random((len(batch_paragraph_strs), self.emb_dim))
            hashes = {}
            uncached_paragraphs = []
            uncached_hashes = []
            for ind, i in enumerate(batch_paragraph_strs):
                hash = self.str2hash(i)
                hashes[hash] = ind
                if hash in self.cache_hash2array:
                    batch_paragraph_array[ind,:] = deepcopy(self.cache_hash2array[hash])
                else:
                    uncached_paragraphs.append(i)
                    uncached_hashes.append(hash)
                    self.cache_hash2str[hash] = i
            inputs = self.tokenizer.batch_encode_plus(
                uncached_paragraphs,
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 max_length=self.max_paragraph_len,
                 return_tensors='pt'
             )
            if len(inputs):
                tmp_device = next(self.bert_paragraph_encoder.parameters()).device
                inputs = {i:inputs[i].to(tmp_device) for i in inputs}
                uncached_paragraph_array = self.bert_paragraph_encoder(
                    **inputs
                ).detach().cpu().numpy()
                for ind, i in enumerate(uncached_paragraph_array):
                    self.cache_hash2array[uncached_hashes[ind]] = deepcopy(i)
                    batch_paragraph_array[ind,:] = deepcopy(i)
            inputs = self.tokenizer.encode_plus(question_str, add_special_tokens=True,
                   max_length=self.max_question_len, pad_to_max_length=True,
                   return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            inputs = {i:inputs[i].to(tmp_device) for i in inputs}
            question_array = self.bert_question_encoder(
                **inputs
            )
            relevance_scores = torch.sigmoid(
                torch.mm(torch.tensor(batch_paragraph_array, dtype=question_array.dtype).to(question_array.device),
                         question_array.T)).reshape(-1)
            rerank_index = torch.argsort(-relevance_scores)
            relevance_scores_numpy = relevance_scores.detach().cpu().numpy()
            rerank_index_numpy = rerank_index.detach().cpu().numpy()
            reranked_paragraphs = [batch_paragraph_strs[i] for i in rerank_index_numpy]
            reranked_relevance_scores = relevance_scores_numpy[rerank_index_numpy]
            return reranked_paragraphs, reranked_relevance_scores, rerank_index_numpy


class RetrieverTrainer(pl.LightningModule):

    def __init__(self, retriever, train_data, dev_data, emb_dim, loss_type, optimizer_type):
        super(RetrieverTrainer, self).__init__()
        self.retriever = retriever
        self.train_data = train_data
        self.dev_data = dev_data
        self.emb_dim = emb_dim
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type

    def forward(self, **kwargs):
        return self.retriever(**kwargs)

    def step_helper(self, batch):
        input_ids_question, attention_mask_question, token_type_ids_question, \
        batch_input_ids_paragraphs, batch_attention_mask_paragraphs, \
        batch_token_type_ids_paragraphs = batch

        inputs = {
            'input_ids_question': input_ids_question,
            'attention_mask_question': attention_mask_question,
            'token_type_ids_question': token_type_ids_question,
            'batch_input_ids_paragraphs': batch_input_ids_paragraphs,
            'batch_attention_mask_paragraphs': batch_attention_mask_paragraphs,
            'batch_token_type_ids_paragraphs': batch_token_type_ids_paragraphs
        }

        h_question, h_paragraphs_batch = self(**inputs)
        batch_size, num_document, emb_dim = batch_input_ids_paragraphs.size()

        all_dots = torch.bmm(h_question.repeat(num_document, 1).unsqueeze(1),
            h_paragraphs_batch.reshape(-1, self.emb_dim).unsqueeze(2)).reshape(batch_size, num_document)
        all_prob = torch.sigmoid(all_dots)

        if self.loss_type == 'negative_sampling':
            pos_loss = - torch.log(all_prob[:, 0]).sum()
            neg_loss = - torch.log(1 - all_prob[:, 1:]).sum()
            loss = pos_loss + neg_loss
        elif self.loss_type == 'classification':
            logits = all_dots
            loss = nn.CrossEntropyLoss()(
                logits, torch.zeros(logits.size()[0], dtype=torch.long).to(logits.device)
            )
        elif self.loss_type == 'triplet_loss':
            # FIXME: for now using only one negative paragraph.
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            loss = triplet_loss(h_question, h_paragraphs_batch[:,0,:], h_paragraphs_batch[:,1,:])
        elif self.loss_type == 'cosine':
            labs = torch.ones(batch_size, num_document)
            labs[:, 1:] *= -1
            labs = labs.reshape(-1).to(h_question.device)
            h_question.repeat(num_document, 1), h_paragraphs_batch.reshape(-1, self.emb_dim)
            loss = torch.nn.CosineEmbeddingLoss()(
                h_question.repeat(num_document, 1), h_paragraphs_batch.reshape(-1, self.emb_dim),
                labs
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

    def validation_step(self, batch, batch_idx):
        loss, all_prob = self.step_helper(batch)
        batch_size = all_prob.size()[0]
        _, y_hat = torch.max(all_prob, 1)
        y_true = torch.zeros(batch_size, dtype=y_hat.dtype).type_as(y_hat)
        val_acc = torch.tensor(accuracy_score(y_true.cpu(), y_hat.cpu())).type_as(y_hat)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        try:
            avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).double().mean()
        except:
            avg_val_loss = torch.cat([x['val_loss'] for x in outputs], 0).mean()
            avg_val_acc = torch.cat([x['val_acc'] for x in outputs], 0).double().mean()

        tqdm_dict = {'val_acc': avg_val_acc, 'val_loss': avg_val_loss}

        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'val_acc': avg_val_acc, 'val_loss': avg_val_loss}
        }
        return results

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