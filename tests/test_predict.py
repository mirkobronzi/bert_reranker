from bert_reranker.data.predict import get_batched_pairs


def test_get_batched_pairs__simple():
    q_pairs = [('q1', ['q1a1, q1a2']), ('q2', ['q2a1, q2a2']), ('q3', ['q3a1, q3a2']),
               ('q4', ['q4a1, q4a2'])]
    result = get_batched_pairs(q_pairs, batch_size=2)
    expected = [[('q1', ['q1a1, q1a2']), ('q2', ['q2a1, q2a2'])],
                [('q3', ['q3a1, q3a2']), ('q4', ['q4a1, q4a2'])]]
    assert expected == result


def test_get_batched_pairs__reminder():
    q_pairs = [('q1', ['q1a1, q1a2']), ('q2', ['q2a1, q2a2']), ('q3', ['q3a1, q3a2'])]
    result = get_batched_pairs(q_pairs, batch_size=2)
    expected = [[('q1', ['q1a1, q1a2']), ('q2', ['q2a1, q2a2'])], [('q3', ['q3a1, q3a2'])]]
    assert expected == result
