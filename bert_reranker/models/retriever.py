import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from bert_reranker.data.data_loader import encode_sentence
from bert_reranker.models.bert_encoder import get_ffw_layers
from bert_reranker.utils.hp_utils import check_and_log_hp

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
        self.softmax = torch.nn.Softmax(dim=0)
        # used for a funny bug/feature of the gradient checkpoint..
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, **kwargs):
        """
        forward method - it will return a score if self.returns_embeddings = False,
                         two embeddings (question/answer) if self.returns_embeddings = True
        """
        raise ValueError('not implemented - use a subclass.')

    def compute_score(self, **kwargs):
        """
        returns a similarity score.
        """
        raise ValueError('not implemented - use a subclass.')

    def compute_embeddings(self, question, passages):

        num_document = len(passages)
        h_question = checkpoint(
            self.bert_question_encoder,
            question['ids'],
            question['am'],
            question['tt'],
            self.dummy_tensor)

        h_paragraph_list = []
        for i in range(num_document):
            h_paragraph = checkpoint(self.bert_paragraph_encoder,
                                     passages[i]['ids'],
                                     passages[i]['am'],
                                     passages[i]['tt'],
                                     self.dummy_tensor)
            h_paragraph_list.append(h_paragraph)

        h_paragraphs_batch = torch.stack(h_paragraph_list, dim=1)

        return h_question, h_paragraphs_batch

    def embed_paragraph(self, paragraph):
        self.eval()
        with torch.no_grad():
            paragraph_inputs = self.tokenizer.encode_plus(
                paragraph, add_special_tokens=True, max_length=self.max_paragraph_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_paragraph_encoder.parameters()).device
            inputs = {k: v.to(tmp_device) for k, v in paragraph_inputs.items()}

            paragraph_embedding = self.bert_paragraph_encoder(**inputs)
        return paragraph_embedding

    def embed_question(self, question):
        self.eval()
        with torch.no_grad():
            question_inputs = self.tokenizer.encode_plus(
                question, add_special_tokens=True, max_length=self.max_question_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            inputs = {k: v.to(tmp_device) for k, v in question_inputs.items()}

            question_embedding = self.bert_question_encoder(**inputs)
        return question_embedding

    def predict(self, question, enc_passages):
        """

        :param question: a string (to encode)
        :param enc_passages: a list of passages (already encoded)
        :return: the prediction (index) and the normalized score.
        """
        self.eval()
        with torch.no_grad():
            # TODO this is only a single batch
            enc_question = encode_sentence(question, self.max_question_len, self.tokenizer)
            enc_question = _add_batch_dim(enc_question)
            enc_passages = [_add_batch_dim(enc_passage) for enc_passage in enc_passages]

            relevance_scores = self.compute_score(question=enc_question, passages=enc_passages)
            relevance_scores = relevance_scores.squeeze(0)  # no batch dimension

            normalized_scores = self.softmax(relevance_scores)
            highest_norm_score, prediction = torch.max(normalized_scores, 0)
            return prediction, highest_norm_score


class EmbeddingRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug):
        super(EmbeddingRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = True

    def forward(self, question, passages):
        return self.compute_embeddings(question, passages)

    def compute_score(self, **kwargs):
        q_emb, p_embs = self.forward(**kwargs)
        return torch.bmm(q_emb.unsqueeze(1), p_embs.transpose(2, 1)).squeeze(1)


def _add_batch_dim(tensor):
    new_tensor = {}
    for k, v in tensor.items():
        new_tensor[k] = torch.unsqueeze(v, 0)
    return new_tensor


class FeedForwardRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
                 max_paragraph_len, debug, model_hyper_params, previous_hidden_size):
        super(FeedForwardRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = False

        check_and_log_hp(['retriever_layer_sizes'], model_hyper_params)
        ffw_layers = get_ffw_layers(
            previous_hidden_size * 2, model_hyper_params['dropout'],
            model_hyper_params['retriever_layer_sizes'] + [1], False)
        self.ffw_net = nn.Sequential(*ffw_layers)

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                batch_token_type_ids_paragraphs):
        q_emb, p_embs = self.compute_embeddings(
            input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs)
        _, n_paragraph, _ = p_embs.shape
        concatenated_embs = torch.cat((q_emb.unsqueeze(1).repeat(1, n_paragraph, 1), p_embs), dim=2)
        logits = self.ffw_net(concatenated_embs)
        return logits.squeeze(dim=2)

    def compute_score(self, **kwargs):
        return self.forward(**kwargs)
