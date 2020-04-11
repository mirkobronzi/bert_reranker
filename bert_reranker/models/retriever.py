import logging
import random
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from bert_reranker.models.optimizer import get_optimizer

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

    def predict(self, question_str: str, batch_paragraph_strs: List[str]):
        self.eval()
        with torch.no_grad():
            # TODO this is only a single batch

            paragraph_inputs = self.tokenizer.batch_encode_plus(
                 batch_paragraph_strs,
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 max_length=self.max_paragraph_len,
                 return_tensors='pt'
             )

            tmp_device = next(self.bert_paragraph_encoder.parameters()).device
            p_inputs = {k: v.to(tmp_device).unsqueeze(0) for k, v in paragraph_inputs.items()}

            question_inputs = self.tokenizer.encode_plus(
                question_str, add_special_tokens=True, max_length=self.max_question_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            q_inputs = {k: v.to(tmp_device) for k, v in question_inputs.items()}

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
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

        self.val_metrics = {}

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
            pos_preds = []
            neg_preds = all_prob.clone()
            for count, target in enumerate(targets):
                pos_preds.append(neg_preds[count][target].clone())
                neg_preds[count][target] = 0
            pos_preds = torch.tensor(pos_preds).to(q_emb.device)
            pos_loss = - torch.log(pos_preds).sum()
            neg_loss = - torch.log(1 - neg_preds).sum()
            loss = pos_loss + neg_loss
        elif self.loss_type == 'classification':
            # all_dots are the logits
            loss = self.cross_entropy(all_dots, targets)
        elif self.loss_type == 'triplet_loss':
            # draw random wrong targets
            wrong_targs = []
            for target in targets:
                wrong_targ = list(range(0, p_embs.shape[1]))
                wrong_targ.remove(target)
                random.shuffle(wrong_targ)
                wrong_targs.extend([wrong_targ[0]])

            pos_pembs = torch.stack(
                [p_embs[idx, target] for (idx, target) in enumerate(targets)])
            neg_pembs = torch.stack(
                [p_embs[idx, wrong_targ] for (idx, wrong_targ) in enumerate(wrong_targs)])

            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            loss = triplet_loss(q_emb, pos_pembs, neg_pembs)
        elif self.loss_type == 'cosine':
            # every target is set to -1 = except for the correct answer (which is 1)
            sin_targets = [[-1] * num_document for _ in range(batch_size)]
            for i, target in enumerate(targets):
                sin_targets[i][target.cpu().item()] = 1

            sin_targets = torch.tensor(sin_targets).reshape(-1).to(q_emb.device)
            q_emb.repeat(num_document, 1), p_embs.reshape(-1, emb_dim)
            loss = self.cosine_loss(
                q_emb.repeat(num_document, 1), p_embs.reshape(-1, emb_dim),
                sin_targets
            )
        else:
            raise ValueError('loss_type {} not supported. Please choose between negative_sampling,'
                             ' classification, cosine')
        return loss, all_prob

    def training_step(self, batch, batch_idx):
        train_loss, _ = self.step_helper(batch)
        # logs
        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def training_step_end(self, outputs):
        loss_value = outputs['loss'].mean()
        tensorboard_logs = {'train_loss': loss_value}
        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataset_number=0):
        # if self.dev_data is a dataloader, there is no provided
        # dataset_number, hence the default value at 0
        loss, all_prob = self.step_helper(batch)
        _, predictions = torch.max(all_prob, 1)
        targets = batch[-1]
        val_acc = torch.tensor(accuracy_score(targets.cpu(), predictions.cpu())).to(targets.device)

        return {'val_loss_' + str(dataset_number): loss, 'val_acc_' + str(dataset_number): val_acc}

    def validation_epoch_end(self, outputs):
        """

        :param outputs: if dev is a single dataloader, then this is an object with 2 dimensions:
                        validation_point, metric_name
                        and it contains N datapoint (tensor with N elements), where N
                        is the number of GPUs.

                        if dev is multiple dataloader, then this is an object with 3 dimensions:
                        dataset, validation_point, metric_name
                        and it contains N datapoint (tensor with N elements), where N
                        is the number of GPUs.
        :return:
        """

        if len(self.dev_data) > 1 and type(self.dev_data) is list:
            # Evaluate all validation sets (if there are more than 1)
            val_metrics = {}
            for dataset_index in range(len(self.dev_data)):
                avg_val_loss = self._comput_mean_for_metric(dataset_index, 'val_loss_', outputs)
                avg_val_acc = self._comput_mean_for_metric(dataset_index, 'val_acc_', outputs)
                val_metrics['val_acc_' + str(dataset_index)] = avg_val_acc
                val_metrics['val_loss_' + str(dataset_index)] = avg_val_loss
        else:  # only one dev set provided
            avg_val_loss = self._comput_mean_for_metric(None, 'val_loss_', outputs)
            avg_val_acc = self._comput_mean_for_metric(None, 'val_acc_', outputs)

            val_metrics = {'val_acc_0': avg_val_acc, 'val_loss_0': avg_val_loss}

        results = {
            'progress_bar': val_metrics,
            'log': val_metrics
        }
        return results

    def _comput_mean_for_metric(self, dataset_index, metric_name, outputs):
        if dataset_index is not None:
            outputs = outputs[dataset_index]
            metric_index = dataset_index
        else:
            metric_index = 0

        datapoints = [x[metric_name + str(metric_index)] for x in outputs]
        if len(datapoints[0].shape) == 0:
            # if just a scalar, create a fake empty dimension for the cat
            datapoints = [dp.unsqueeze(0) for dp in datapoints]
        val_losses = torch.cat(datapoints)
        avg_val_loss = val_losses.mean()
        return avg_val_loss

    def test_step(self, batch, batch_idx):
        # we do the same stuff as in the validation phase
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return get_optimizer(self.optimizer_type, self.retriever)

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.dev_data

    def test_dataloader(self):
        return self.test_data
