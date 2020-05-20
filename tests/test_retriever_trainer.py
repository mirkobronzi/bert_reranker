import torch

from bert_reranker.models.retriever_trainer import soft_cross_entropy


def test_soft_cross_entropy__simple():
    # kind of fake test - checking soft-targets with same weight will push the logits to be equal
    # i.e., case 5, 5 in the example below (which has index 5)
    losses = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        logits = torch.tensor([i, 10 - i], dtype=torch.float32)
        soft_targets = torch.tensor([0.5, 0.5])
        loss = soft_cross_entropy(logits, soft_targets)
        losses.append(loss)
    assert torch.stack(losses).argmin(0) == 5
