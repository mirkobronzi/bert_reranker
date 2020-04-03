import logging
import ntpath
import os
import random

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import json

logger = logging.getLogger(__name__)


def remove_html_toks(s):
    html_toks = [
        '<P>',
        '</P>',
        '<H1>',
        '</H1>',
        '</H2>',
        '</H2>',
    ]
    for i in html_toks:
        s = s.replace(i, '')
    return s


def shuffle_paragraphs(paragraphs):
    assert len(paragraphs) == 3
    random_indices = list(range(3))
    random.shuffle(random_indices)
    shuffled_paragraphs = [paragraphs[i] for i in random_indices]
    target = random_indices.index(0)
    return shuffled_paragraphs, target


def json_to_dataset(json_file, max_question_len, max_paragraph_len, tokenizer):
    """
    output format is a dataset with a question ([batch_size, seq_len])
    and a set of paragraphs ([batch_size, paragraph_size, seq_len]).

    A dataloader contains the token_ids, the padding, and the bert token type.


    :param natq_json_file:
    :param cache_folder:
    :param max_question_len:
    :param max_paragraph_len:
    :param tokenizer:
    :return:
    """

    if not os.path.exists(json_file):
        raise Exception('{} not found'.format(json_file))

    with open(json_file, 'r', encoding='utf-8', errors='ignore') as in_stream:

        qa_pairs = json.load(in_stream)
        input_ids_question = []
        attention_mask_question = []
        token_type_ids_question = []
        batch_input_ids_paragraphs = []
        batch_attention_mask_paragraphs = []
        batch_token_type_ids_paragraphs = []
        targets = []

        for question, answers in tqdm(qa_pairs):

                paragraphs = [remove_html_toks(i) for i in answers]

                shuffled_paragraphs, target = shuffle_paragraphs(paragraphs)

                input_question = tokenizer.encode_plus(question, add_special_tokens=True,
                                                       max_length=max_question_len, pad_to_max_length=True,
                                                       return_tensors='pt')
                inputs_paragraph = tokenizer.batch_encode_plus(shuffled_paragraphs,
                                                               add_special_tokens=True,
                                                               pad_to_max_length=True,
                                                               max_length=max_paragraph_len,
                                                               return_tensors='pt'
                                                               )
                input_ids_question.append(input_question['input_ids'])
                attention_mask_question.append(input_question['attention_mask'])
                token_type_ids_question.append(input_question['token_type_ids'])
                batch_input_ids_paragraphs.append(inputs_paragraph['input_ids'].unsqueeze(0))
                batch_attention_mask_paragraphs.append(inputs_paragraph['attention_mask'].unsqueeze(0))
                batch_token_type_ids_paragraphs.append(inputs_paragraph['token_type_ids'].unsqueeze(0))
                targets.append(target)

        dataset = TensorDataset(
            torch.cat(input_ids_question),
            torch.cat(attention_mask_question),
            torch.cat(token_type_ids_question),
            torch.cat(batch_input_ids_paragraphs),
            torch.cat(batch_attention_mask_paragraphs),
            torch.cat(batch_token_type_ids_paragraphs),
            torch.tensor(targets)
        )

    return dataset


def generate_dataloaders(train_file, dev_file, cache_folder, max_question_len,
                         max_paragraph_len, tokenizer, batch_size):

    train_file_name = ntpath.basename(train_file)
    dev_file_name = ntpath.basename(dev_file)
    cached_train = os.path.join(cache_folder, train_file_name + '.pt')
    cached_dev = os.path.join(cache_folder, dev_file_name + '.pt')

    if not os.path.exists(cached_train):
        logger.info('cached file {} not found - computing it'.format(cached_train))
        train_dataset = json_to_dataset(train_file, max_question_len, max_paragraph_len,
                                        tokenizer)
        torch.save(train_dataset, cached_train)
    else:
        logger.info('cached file {} found - loading'.format(cached_train))

    if not os.path.exists(cached_dev):
        logger.info('cached file {} not found - computing it'.format(cached_dev))
        dev_dataset = json_to_dataset(dev_file, max_question_len, max_paragraph_len,
                                      tokenizer)
        torch.save(dev_dataset, cached_dev)
    else:
        logger.info('cached file {} found - loading'.format(cached_dev))

    train_set = torch.load(cached_train)
    dev_set = torch.load(cached_dev)

    logger.info('train size {} / dev size {}'.format(len(train_set), len(dev_set)))

    return (DataLoader(train_set, batch_size=batch_size),
            DataLoader(dev_set, batch_size=batch_size))
