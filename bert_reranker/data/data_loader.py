import json
import logging
import os
import random
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def shuffle_paragraphs(paragraphs):
    n_paragraph = len(paragraphs)
    random_indices = list(range(n_paragraph))
    random.shuffle(random_indices)
    shuffled_paragraphs = [paragraphs[i] for i in random_indices]
    target = random_indices.index(0)
    return shuffled_paragraphs, target


def encode_sentence(sentence, max_length, tokenizer):
    input_question = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
    return {'ids': input_question['input_ids'].squeeze(0),
            'am': input_question['attention_mask'].squeeze(0),
            'tt': input_question['token_type_ids'].squeeze(0)}


def _get_passages_by_source(json_data):
    source2passages = defaultdict(list)
    passage_id2source = {}
    passage_id2index = {}
    for passage in json_data['passages']:
        source = passage['source']
        source2passages[source].append(passage['reference']['section_headers'][0])
        passage_id = passage['passage_id']
        if passage_id in passage_id2source:
            raise ValueError('duplicate passage id: {}'.format(passage_id))
        passage_id2source[passage_id] = source
        passage_id2index[passage_id] = len(source2passages[source]) - 1
    return source2passages, passage_id2source, passage_id2index


def _encode_passages(source2passages, max_passage_length, tokenizer):
    encoded_source2passages = defaultdict(list)
    for source, passages in source2passages.items():
        for passage in passages:
            encoded_passage = encode_sentence(passage, max_passage_length, tokenizer)
            encoded_source2passages[source].append(encoded_passage)
    return encoded_source2passages


class ReRankerDataset(Dataset):

    def __init__(self, json_file, max_example_len, max_passage_len, tokenizer):
        self.max_example_len = max_example_len
        self.max_passage_len = max_passage_len
        self.tokenizer = tokenizer

        if not os.path.exists(json_file):
            raise Exception('{} not found'.format(json_file))

        with open(json_file, 'r', encoding='utf-8') as in_stream:
            self.json_data = json.load(in_stream)

        source2passages, self.passage_id2source, self.passage_id2index = _get_passages_by_source(
            self.json_data)
        self.encoded_source2passages = _encode_passages(
            source2passages, max_passage_len, tokenizer)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        example = self.json_data['examples'][idx]
        question = example['question']
        passage_id = example['passage_id']  # this is our target
        encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)
        source = self.passage_id2source[passage_id]
        # this is the index of the target in the list of passages for the current source
        target_idx = self.passage_id2index[passage_id]
        return {'question': encoded_question, 'target_idx': target_idx,
                'passages': self.encoded_source2passages[source]}


def generate_dataloader(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    dataset = ReRankerDataset(data_file, max_question_len, max_paragraph_len, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
