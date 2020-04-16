import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from bert_reranker.models.optimizer import get_optimizer


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

        if loss_type == 'classification':
            self.cross_entropy = nn.CrossEntropyLoss()

        if self.loss_type == 'cosine':
            self.cosine_loss = torch.nn.CosineEmbeddingLoss()

        self.val_metrics = {}

    def forward(self, **kwargs):
        return self.retriever(**kwargs)

    def step_helper(self, batch):
        inputs = {k: v for k, v in batch.items() if k != 'target'}
        targets = batch['target']

        if self.loss_type == 'negative_sampling':
            if self.retriever.returns_embeddings:
                q_emb, p_embs = self.retriever(**inputs)
                logits = torch.bmm(q_emb.unsqueeze(1), p_embs.transpose(2, 1)).squeeze(1)
            else:
                logits = self.retriever(**inputs)
            all_prob = torch.sigmoid(logits)

            pos_preds = []
            neg_preds = all_prob.clone()
            for count, target in enumerate(targets):
                pos_preds.append(neg_preds[count][target].clone())
                neg_preds[count][target] = 0
            pos_preds = torch.tensor(pos_preds).to(all_prob.device)
            pos_loss = - torch.log(pos_preds).sum()
            neg_loss = - torch.log(1 - neg_preds).sum()
            loss = pos_loss + neg_loss
        elif self.loss_type == 'classification':
            if self.retriever.returns_embeddings:
                q_emb, p_embs = self.retriever(**inputs)
                logits = torch.bmm(q_emb.unsqueeze(1), p_embs.transpose(2, 1)).squeeze(1)
            else:
                logits = self.retriever(**inputs)

            loss = self.cross_entropy(logits, targets)
        elif self.loss_type == 'triplet_loss':
            if not self.retriever.returns_embeddings:
                raise ValueError('triples_loss is compatible only with models returning embeddings')
            q_emb, p_embs = self.retriever(**inputs)

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
            # to compute final probabilities
            logits = torch.bmm(q_emb.unsqueeze(1), p_embs.transpose(2, 1)).squeeze(1)
        elif self.loss_type == 'cosine':
            if not self.retriever.returns_embeddings:
                raise ValueError('triples_loss is compatible only with models returning embeddings')
            q_emb, p_embs = self.retriever(**inputs)
            batch_size, num_document, emb_dim = p_embs.size()

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
            # to compute final probabilities
            logits = torch.bmm(q_emb.unsqueeze(1), p_embs.transpose(2, 1)).squeeze(1)
        else:
            raise ValueError('loss_type {} not supported. Please choose between negative_sampling,'
                             ' classification, cosine, triplet_loss')
        all_prob = torch.sigmoid(logits)
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
        targets = batch['target']
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

        #  dev_data can be either a single dataloader, or a list of dataloaders
        #  for evaluation on many test sets

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
