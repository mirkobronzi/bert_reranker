import logging
from typing import List

import torch
import torch.nn as nn

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


class EmbeddingRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug):
        super(EmbeddingRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                batch_token_type_ids_paragraphs):
        return self.compute_emebeddings(
            input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs)
