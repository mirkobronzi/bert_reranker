import json
import logging
import os
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def encode_sentence(sentence, max_length, tokenizer):
    input_question = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
    return {'ids': input_question['input_ids'].squeeze(0),
            'am': input_question['attention_mask'].squeeze(0),
            'tt': input_question['token_type_ids'].squeeze(0)}


def get_passages_by_source(json_data):
    source2passages = defaultdict(list)
    pid2passage = {}
    pid2index = {}
    for passage in json_data['passages']:
        source = passage['source']
        passage_id = passage['passage_id']

        source2passages[source].append(passage)
        if passage_id in pid2passage:
            raise ValueError('duplicate passage id: {}'.format(passage_id))

        if is_in_distribution(passage):
            pid2index[passage_id] = len(source2passages[source]) - 1
        else:
            pid2index[passage_id] = -1

        pid2passage[passage_id] = passage

    return source2passages, pid2passage, pid2index


def is_in_distribution(passage):
    reference_type = passage['reference_type']
    return reference_type.lower().startswith('faq')


def _encode_passages(source2passages, max_passage_length, tokenizer):
    """
    note - this will only encode in-distribution passages.
    :param source2passages:
    :param max_passage_length:
    :param tokenizer:
    :return:
    """
    source2encoded_passages = defaultdict(list)
    source2id = defaultdict(int)
    source2ood = defaultdict(int)
    for source, passages in source2passages.items():
        for passage in passages:
            if is_in_distribution(passage):
                passage_text = get_passage_last_header(passage)
                encoded_passage = encode_sentence(passage_text, max_passage_length, tokenizer)
                source2encoded_passages[source].append(encoded_passage)
                source2id[source] += 1
            else:
                source2ood[source] += 1
    return source2encoded_passages, source2id, source2ood


def get_passage_last_header(passage, return_error_for_ood=False):
    if is_in_distribution(passage):
        return passage['reference']['section_headers'][0]
    elif return_error_for_ood:
        raise ValueError('passage is ood')
    else:
        return '__out-of-distribution__'


def get_question(example):
    return example['question']


def get_passage_id(example):
    return example['passage_id']


class ReRankerDataset(Dataset):

    def __init__(self, json_file, max_example_len, max_passage_len, tokenizer):
        self.max_example_len = max_example_len
        self.max_passage_len = max_passage_len
        self.tokenizer = tokenizer

        if not os.path.exists(json_file):
            raise Exception('{} not found'.format(json_file))

        with open(json_file, 'r', encoding='utf-8') as in_stream:
            json_data = json.load(in_stream)

        source2passages, pid2passage, pid2index = get_passages_by_source(
            json_data)

        self.encoded_source2passages, source2id, source2ood = _encode_passages(
            source2passages, max_passage_len, tokenizer)
        self.pid2passage = pid2passage
        self.pid2index = pid2index
        self.examples = json_data['examples']
        logger.info('loaded passages from file {} - found {} sources'.format(
            json_file, len(source2id)))
        for source in source2id.keys():
            logger.info('source "{}": found {} in-distribution and {} out-of-distribution'.format(
                source, source2id[source], source2ood[source]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        question = get_question(example)
        passage_id = example['passage_id']  # this is the related passage
        encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)

        passage = self.pid2passage[passage_id]
        # this is the index of the target in the list of passages for the current source
        if is_in_distribution(passage):
            target_idx = self.pid2index[passage_id]
        else:
            target_idx = -1
        source = self.pid2passage[passage_id]['source']
        return {'question': encoded_question, 'target_idx': target_idx,
                'passages': self.encoded_source2passages[source]}


def generate_dataloader(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    dataset = ReRankerDataset(data_file, max_question_len, max_paragraph_len, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
