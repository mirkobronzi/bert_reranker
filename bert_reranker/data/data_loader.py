import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import json


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


def natq_to_dataset(natq_file, cache_folder, max_question_len, max_paragraph_len, tokenizer):
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

    if not os.path.exists(natq_file):
        raise Exception('{} not found'.format(natq_file))

    with open(natq_file, 'r', encoding='utf-8', errors='ignore') as in_stream:

        qa_pairs = json.load(in_stream)
        input_ids_question, attention_mask_question, token_type_ids_question, \
        batch_input_ids_paragraphs, batch_attention_mask_paragraphs, \
        batch_token_type_ids_paragraphs = [], [], [], [], [], []

        for question, answers in tqdm(qa_pairs):

                paras = [remove_html_toks(i) for i in answers]

                input_question = tokenizer.encode_plus(question, add_special_tokens=True,
                                                       max_length=max_question_len, pad_to_max_length=True,
                                                       return_tensors='pt')
                inputs_paragraph = tokenizer.batch_encode_plus(paras,
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

        dataset = TensorDataset(
            torch.cat(input_ids_question),
            torch.cat(attention_mask_question),
            torch.cat(token_type_ids_question),
            torch.cat(batch_input_ids_paragraphs),
            torch.cat(batch_attention_mask_paragraphs),
            torch.cat(batch_token_type_ids_paragraphs),
        )

    return dataset


def generate_natq_dataloaders(natq_train_file, natq_dev_file, cache_folder, max_question_len, max_paragraph_len,
                              tokenizer, batch_size):

    cached_train = os.path.join(cache_folder, 'natq_train.pt')
    cached_dev = os.path.join(cache_folder, 'natq_dev.pt')

    if (not os.path.exists(cached_train)):
        train_dataset = natq_to_dataset(natq_train_file, cache_folder, max_question_len, max_paragraph_len, tokenizer)
        torch.save(train_dataset, cached_train)

    if (not os.path.exists(cached_dev)):
        dev_dataset = natq_to_dataset(natq_dev_file, cache_folder, max_question_len, max_paragraph_len, tokenizer)
        torch.save(dev_dataset, cached_dev)

    train_set = torch.load(cached_train)
    dev_set = torch.load(cached_dev)
    return (DataLoader(train_set, batch_size=batch_size),
            DataLoader(dev_set, batch_size=batch_size))


def generate_fake_dataloaders(num_dat, batch_size, max_question_len, max_paragraph_len,
                              tokenizer):
    ## convert things to data loaders
    txt = 'I am a question'
    input_question = tokenizer.encode_plus(txt, add_special_tokens=True,
                                           max_length=max_question_len, pad_to_max_length=True,
                                           return_tensors='pt')
    inputs_paragraph = tokenizer.batch_encode_plus(['I am positve' * 3, 'I am negative' * 4, 'I am negative', 'I am negative super'],
                                                   add_special_tokens=True,
                                                   pad_to_max_length=True,
                                                   max_length=max_paragraph_len,
                                                   return_tensors='pt'
                                                   )
    dataset = TensorDataset(
        input_question['input_ids'].repeat(num_dat, 1),
        input_question['attention_mask'].repeat(num_dat, 1),
        input_question['token_type_ids'].repeat(num_dat, 1),
        inputs_paragraph['input_ids'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['attention_mask'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['token_type_ids'].unsqueeze(0).repeat(num_dat, 1, 1)
    )

    dataset_dev = TensorDataset(
        input_question['input_ids'].repeat(num_dat, 1),
        input_question['attention_mask'].repeat(num_dat, 1),
        input_question['token_type_ids'].repeat(num_dat, 1),
        inputs_paragraph['input_ids'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['attention_mask'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['token_type_ids'].unsqueeze(0).repeat(num_dat, 1, 1)
    )

    return DataLoader(dataset, batch_size=batch_size), DataLoader(dataset_dev, batch_size=batch_size)


