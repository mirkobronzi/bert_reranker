import logging

import torch

logger = logging.getLogger(__name__)


def get_optimizer(optimizer, model):

    optimizer_name = optimizer['name']

    if optimizer_name == 'adamw':
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    elif optimizer_name == 'adam':
        layer_lr = optimizer['lr']
        logger.info('using adam with lr={}'.format(layer_lr))
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=layer_lr)
    elif optimizer_name == 'adam_diff_lr':
        ffw_lr = optimizer['ffw_lr']
        bert_lrs = optimizer['bert_lrs']
        logger.info('ffw lr={} / bert layer lrs={}'.format(ffw_lr, bert_lrs))
        lsr = [
            {'params': _get_grad_params(model.bert_question_encoder.net.parameters()),
             'lr': ffw_lr},
            {'params': _get_grad_params(model.bert_paragraph_encoder.net.parameters()),
             'lr': ffw_lr}]
        for i in range(12):
            layer_lr = bert_lrs[i]
            lsr.append(
                {'params':_get_grad_params(model.bert_question_encoder.bert.encoder.layer[i].parameters()),
                 'lr': layer_lr})
            lsr.append(
                {'params': _get_grad_params(model.bert_paragraph_encoder.bert.encoder.layer[i].parameters()),
                 'lr': layer_lr})
        opt = torch.optim.Adam(lsr)
        return opt
    else:
        raise ValueError('optimizer {} not supported'.format(optimizer_name))


def _get_grad_params(model_params):
    return [p for p in model_params if p.requires_grad]