import logging
from typing import List

import torch
import torch.nn as nn

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
        self.cache_hash2str = {}
        self.cache_hash2array = {}

    def forward(self, **kwargs):
        raise ValueError('not implemented - use a subclass.')

    def compute_emebeddings(
            self, input_ids_question, attention_mask_question, token_type_ids_question,
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

        h_paragraph_list = []
        for i in range(num_document):
            h_paragraphs = self.bert_paragraph_encoder(
                input_ids=batch_input_ids_paragraphs[:, i, :],
                attention_mask=batch_attention_mask_paragraphs[:, i, :],
                token_type_ids=batch_token_type_ids_paragraphs[:, i, :])
            h_paragraph_list.append(h_paragraphs)
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
            relevance_scores = torch.matmul(q_emb, p_embs.squeeze(0).T).squeeze(0)

            rerank_index = torch.argsort(-relevance_scores)
            relevance_scores_numpy = relevance_scores.detach().cpu().numpy()
            rerank_index_numpy = rerank_index.detach().cpu().numpy()
            reranked_paragraphs = [batch_paragraph_strs[i] for i in rerank_index_numpy]
            reranked_relevance_scores = relevance_scores_numpy[rerank_index_numpy]
            return reranked_paragraphs, reranked_relevance_scores, rerank_index_numpy


class EmbeddingRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug):
        super(EmbeddingRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = True

    def forward(self, q_ids, q_am, q_tt, p_ids, p_am, p_tt):
        return self.compute_emebeddings(q_ids, q_am, q_tt, p_ids, p_am, p_tt)


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
        q_emb, p_embs = self.compute_emebeddings(
            input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs)
        _, n_paragraph, _ = p_embs.shape
        concatenated_embs = torch.cat((q_emb.unsqueeze(1).repeat(1, n_paragraph, 1), p_embs), dim=2)
        logits = self.ffw_net(concatenated_embs)
        return logits.squeeze(dim=2)
