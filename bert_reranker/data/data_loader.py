import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import json


TRAIN_NAME = 'natq_train.pt'
DEV_NAME = 'natq_dev.pt' \
           ''

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


def process_natq_clean(natq_json_file, cache_folder, max_question_len, max_paragraph_len,
                       tokenizer):

    if not os.path.exists(natq_json_file):
        raise Exception('{} not found'.format(natq_json_file))

    cached_train = os.path.join(cache_folder, TRAIN_NAME)
    train_exists = os.path.exists(cached_train)
    cached_dev = os.path.join(cache_folder, DEV_NAME)
    dev_exists = os.path.exists(cached_dev)

    if train_exists and dev_exists:
        return

    if not train_exists:
        train_input_ids_question, train_attention_mask_question, train_token_type_ids_question, \
        train_batch_input_ids_paragraphs, train_batch_attention_mask_paragraphs, \
        train_batch_token_type_ids_paragraphs = [], [], [], [], [], []
    if not dev_exists:
        dev_input_ids_question, dev_attention_mask_question, dev_token_type_ids_question, \
        dev_batch_input_ids_paragraphs, dev_batch_attention_mask_paragraphs, \
        dev_batch_token_type_ids_paragraphs = [], [], [], [], [], []

    with open(natq_json_file, 'r', encoding='utf-8', errors='ignore') as f:
        for l in tqdm(f):
            d = json.loads(l)

            if d['num_positives'] >= 1 and d['num_negatives'] >= 2:

                if d['dataset'] == 'train' and train_exists:
                    continue
                if d['dataset'] == 'dev' and dev_exists:
                    continue

                q = d['question']
                paras = d['right_paragraphs'][:1] + d['wrong_paragraphs'][:2]
                paras = [remove_html_toks(i) for i in paras]

                input_question = tokenizer.encode_plus(q, add_special_tokens=True,
                                                       max_length=max_question_len, pad_to_max_length=True,
                                                       return_tensors='pt')
                inputs_paragraph = tokenizer.batch_encode_plus(paras,
                                                               add_special_tokens=True,
                                                               pad_to_max_length=True,
                                                               max_length=max_paragraph_len,
                                                               return_tensors='pt'
                                                               )

                if d['dataset'] == 'train':
                    train_input_ids_question.append(input_question['input_ids'])
                    train_attention_mask_question.append(input_question['attention_mask'])
                    train_token_type_ids_question.append(input_question['token_type_ids'])
                    train_batch_input_ids_paragraphs.append(inputs_paragraph['input_ids'].unsqueeze(0))
                    train_batch_attention_mask_paragraphs.append(inputs_paragraph['attention_mask'].unsqueeze(0))
                    train_batch_token_type_ids_paragraphs.append(inputs_paragraph['token_type_ids'].unsqueeze(0))

                elif d['dataset'] == 'dev':
                    dev_input_ids_question.append(input_question['input_ids'])
                    dev_attention_mask_question.append(input_question['attention_mask'])
                    dev_token_type_ids_question.append(input_question['token_type_ids'])
                    dev_batch_input_ids_paragraphs.append(inputs_paragraph['input_ids'].unsqueeze(0))
                    dev_batch_attention_mask_paragraphs.append(inputs_paragraph['attention_mask'].unsqueeze(0))
                    dev_batch_token_type_ids_paragraphs.append(inputs_paragraph['token_type_ids'].unsqueeze(0))
    if not dev_exists:
        dev_set = TensorDataset(
            torch.cat(dev_input_ids_question),
            torch.cat(dev_attention_mask_question),
            torch.cat(dev_token_type_ids_question),
            torch.cat(dev_batch_input_ids_paragraphs),
            torch.cat(dev_batch_attention_mask_paragraphs),
            torch.cat(dev_batch_token_type_ids_paragraphs),
        )
        torch.save(dev_set, cached_dev)

    if not train_exists:
        train_set = TensorDataset(
            torch.cat(train_input_ids_question),
            torch.cat(train_attention_mask_question),
            torch.cat(train_token_type_ids_question),
            torch.cat(train_batch_input_ids_paragraphs),
            torch.cat(train_batch_attention_mask_paragraphs),
            torch.cat(train_batch_token_type_ids_paragraphs),
        )

        torch.save(train_set, cached_train)


def generate_natq_dataloaders(natq_json_file, cache_folder, max_question_len, max_paragraph_len,
                              tokenizer, batch_size):

    cached_train = os.path.join(cache_folder, 'natq_train.pt')
    cached_dev = os.path.join(cache_folder, 'natq_dev.pt')
    if (not os.path.exists(cached_train)) or (not os.path.exists(cached_dev)):
        process_natq_clean(natq_json_file, cache_folder, max_question_len, max_paragraph_len,
                           tokenizer)

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


