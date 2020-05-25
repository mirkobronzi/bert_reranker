import torch

from bert_reranker.models.retriever_trainer import soft_cross_entropy, prepare_soft_targets


def test_soft_cross_entropy__simple():
    # kind of fake test - checking soft-targets with same weight will push the logits to be equal
    # i.e., case 5, 5 in the example below (which has index 5)
    losses = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        logits = torch.tensor([[i, 10 - i]], dtype=torch.float32)
        soft_targets = torch.tensor([0.5, 0.5])
        loss = soft_cross_entropy(logits, soft_targets).squeeze(0)
        losses.append(loss)
    assert torch.stack(losses).argmin(0) == 5


def test_prepare_soft_targets__simple():
    num_classes = 4
    target_ints = torch.tensor([0, 1, -1])
    soft_targets = prepare_soft_targets(target_ints, num_classes)
    expected = torch.tensor([
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500]], dtype=torch.float64)
    assert torch.all(expected.eq(soft_targets))
