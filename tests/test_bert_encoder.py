import numpy
from torch import tensor

from bert_reranker.models.general_encoder import compute_average_with_padding


def test_compute_average_with_padding__simple():
    padding = tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
    # batch x sentence x emb_size
    tens = tensor([[[0.2, 0.2], [0.3, 0.4], [0.5, 0.6]],
                   [[0.3, 0.6], [0.1, 0.1], [0.8, 0.8]],
                   [[0.3, 0.6], [0.1, 0.1], [0.8, 0.9]],
                   [[0.3, 0.6], [0.1, 0.1], [0.8, 0.9]]])
    # batch x emb_size
    expected = tensor([[0.25, 0.3],
                       [0.4, 0.5],
                       [0.3, 0.6],
                       [0.2, 0.35]])
    result = compute_average_with_padding(tens, padding)
    assert numpy.allclose(expected.numpy(), result.numpy())
