import argparse
import logging
import json

from transformers import AutoTokenizer

from bert_reranker.data.data_loader import clean_text

logger = logging.getLogger(__name__)

def count_cutoff_sentences(sentences, tokenizer, max_length):

    cutoff_results = {}

    n_sentences_cutoff = 0
    original_sentences = []
    cutoff_sentences = []
    for sentence in sentences:

        encoded_sentence = tokenizer.encode(sentence, max_length=max_length, add_special_tokens=True, pad_to_max_length=True)
        decoded_sentence = tokenizer.decode(encoded_sentence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if not clean_text(sentence) == clean_text(decoded_sentence):
            n_sentences_cutoff += 1
            original_sentences.append(clean_text(sentence))
            cutoff_sentences.append(clean_text(decoded_sentence))


    cutoff_results = {
        'n_sentences_cutoff': n_sentences_cutoff,
        'original_questions': original_sentences,
        'cutoff_questions': cutoff_sentences,
        'total_sentences': len(sentences),
        }

    return cutoff_results

def evaluate_tokenizer_cutoff(file_to_evaluate, tokenizer, max_question_length, max_answer_length):
    '''evaluate how much questions are being cutoff based on tokenizer's max length'''
    with open(file_to_evaluate, 'r', encoding='utf-8') as in_stream:
        qa_pairs = json.load(in_stream)

    # Collect all unique questions and answers
    all_questions = []
    all_answers = set()
    for qa_pair in qa_pairs:
        question, answers = qa_pair
        all_questions.append(question)
        all_answers |= set(answers)
    all_answers = list(all_answers)

    # Analyze how much is being cutoff
    cutoff_results_questions = count_cutoff_sentences(all_questions, tokenizer, max_question_length)
    cutoff_results_answers = count_cutoff_sentences(all_answers, tokenizer, max_answer_length)
    cutoff_results = {
        'questions': cutoff_results_questions,
        'answers': cutoff_results_answers,
    }

    logger.info('Max length used for questions: {}'.format(max_question_length))
    logger.info('Number of questions cutoff by tokenizer: {} / {}, ({:3.2f} %)'.format(
        cutoff_results['questions']['n_sentences_cutoff'],
        cutoff_results['questions']['total_sentences'],
        cutoff_results['questions']['n_sentences_cutoff'] / cutoff_results['questions']['total_sentences']*100,
        )
    )
    logger.info('Max length used for answers: {}'.format(max_answer_length))
    logger.info('Number of answers cutoff by tokenizer: {} / {}, ({:3.2f} %)'.format(
        cutoff_results['answers']['n_sentences_cutoff'],
        cutoff_results['answers']['total_sentences'],
        cutoff_results['answers']['n_sentences_cutoff'] / cutoff_results['questions']['total_sentences']*100,
        )
    )

    return cutoff_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        help='data file containing questions and answers', required=True)

    parser.add_argument('--tokenizer_name',
                        help='name of the tokenizer to use', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    max_lengths = [10, 30, 50, 100]

    for max_length in max_lengths:
        cutoff_results =  evaluate_tokenizer_cutoff(args.data_file, tokenizer, max_length, max_length)


if __name__ == '__main__':
    main()
