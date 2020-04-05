import logging

import torch

logger = logging.getLogger(__name__)


def get_optimizer(optimizer_type, model):
    optimizer_args = optimizer_type.split(',')
    optimizer_name = optimizer_args[0]
    optimizer_params = {}
    for entry in optimizer_args[1:]:
        k, v = entry.split('=')
        optimizer_params[k] = v

    if optimizer_name == 'adamw':
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    elif optimizer_name == 'adam':
        lr = optimizer_params.get('lr', 1e-3)
        logger.info('using adam with lr={}'.format(lr))
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    else:
        raise ValueError('optimizer {} not supported'.format(optimizer_type))