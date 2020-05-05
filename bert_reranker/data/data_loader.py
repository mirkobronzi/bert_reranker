import json
import logging
import os
import re
import random
import string

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def remove_links(s):
    '''swap a link with a _URL_ token'''
    return re.sub(r"http\S+", "_URL_", s)


def remove_html_tags(s):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', s)


def remove_standard_punctuation(s):
    punctuation = '¿!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~' # We are leaving out _ for the _URL_ token
    return s.translate(str.maketrans('', '', punctuation))


def remove_extra_whitespace(s):
    s = ' '.join(s.split())
    return s

def clean_text(s):
    s = s.lower()
    s = remove_links(s)
    s = remove_html_tags(s)
    s = remove_standard_punctuation(s)
    s = remove_extra_whitespace(s)
    return s


def shuffle_paragraphs(paragraphs):
    n_paragraph = len(paragraphs)
    random_indices = list(range(n_paragraph))
    random.shuffle(random_indices)
    shuffled_paragraphs = [paragraphs[i] for i in random_indices]
    target = random_indices.index(0)
    return shuffled_paragraphs, target


class ReRankerDataset(Dataset):

    def __init__(self, json_file, max_question_len, max_paragraph_len, tokenizer):
        self.max_question_len = max_question_len
        self.max_paragraph_len = max_paragraph_len
        self.tokenizer = tokenizer

        if not os.path.exists(json_file):
            raise Exception('{} not found'.format(json_file))

        with open(json_file, 'r', encoding='utf-8') as in_stream:
            self.qa_pairs = json.load(in_stream)

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        data = json_entry_to_dataset(
            self.qa_pairs[idx], self.max_question_len, self.max_paragraph_len, self.tokenizer)
        return data


def json_entry_to_dataset(qa_pair, max_question_len, max_paragraph_len, tokenizer):
    question, answers = qa_pair

    paragraphs = [clean_text(i) for i in answers]
    shuffled_paragraphs, target = shuffle_paragraphs(paragraphs)
    input_question = tokenizer.encode_plus(question, add_special_tokens=True,
                                           max_length=max_question_len,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
    inputs_paragraph = tokenizer.batch_encode_plus(shuffled_paragraphs,
                                                   add_special_tokens=True,
                                                   pad_to_max_length=True,
                                                   max_length=max_paragraph_len,
                                                   return_tensors='pt')

    return {'q_ids': input_question['input_ids'].squeeze(0),
            'q_am': input_question['attention_mask'].squeeze(0),
            'q_tt': input_question['token_type_ids'].squeeze(0),
            'p_ids': inputs_paragraph['input_ids'].squeeze(0),
            'p_am': inputs_paragraph['attention_mask'].squeeze(0),
            'p_tt': inputs_paragraph['token_type_ids'].squeeze(0),
            'target': target}


def generate_dataloader(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    dataset = ReRankerDataset(data_file, max_question_len, max_paragraph_len, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
