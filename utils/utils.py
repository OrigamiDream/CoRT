import os
import time
import random
import logging
import argparse
import contextlib
import numpy as np
import tensorflow as tf

from cort.config import Config
from cort.pretrained import migrator, tokenization
from transformers import AutoConfig, AutoTokenizer


DISALLOWED_TOKENS = ['<unk>', '<s>', '</s>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
DISALLOWED_TOKENS += ['[unused{}]'.format(i + 1) for i in range(200)]


@contextlib.contextmanager
def empty_context_manager():
    yield None


def generate_random_id(length=8):
    chars = 'abcdefghijklnmopqrstuvwxyz0123456789'
    return ''.join([random.choice(chars) for _ in range(length)])


def restrict_gpus(config: Config):
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        logging.warning('No available GPUs')
        return

    if config.gpu != 'all':
        desired_gpu = gpus[int(config.gpu)]
        tf.config.set_visible_devices(desired_gpu, 'GPU')
        logging.info('Restricting GPU as /device:GPU:{}'.format(config.gpu))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)


def create_tokenizer_from_config(config):
    if config.model_name == 'korscibert':
        tokenizer = tokenization.create_tokenizer(config.korscibert_vocab, tokenizer_type='bert')
    elif config.model_name == 'korscielectra':
        tokenizer = tokenization.create_tokenizer(config.korscielectra_vocab, tokenizer_type='electra')
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer


def parse_vocabulary(vocab_filepath):
    with open(vocab_filepath, 'r') as f:
        texts = f.read().splitlines()
        vocab = {text: index for index, text in enumerate(texts) if len(text) > 0}
    return vocab


def current_milliseconds():
    return round(time.time() * 1000)


def format_minutes_and_seconds(milliseconds):
    minutes = int(milliseconds / 1000 / 60)
    seconds = int(milliseconds / 1000) - (minutes * 60)
    return minutes, seconds


def parse_pretrained_config(config):
    if config.model_name == 'korscibert':
        vocab = parse_vocabulary(config.korscibert_vocab)
        pretrained_config = migrator.create_base_bert_config(vocab)
    elif config.model_name == 'korscielectra':
        vocab = parse_vocabulary(config.korscielectra_vocab)
        pretrained_config = migrator.create_base_electra_config(vocab)
    else:
        pretrained_config = AutoConfig.from_pretrained(config.model_name)

    pretrained_config.max_position_embeddings = min(512, pretrained_config.max_position_embeddings)
    return pretrained_config


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    config = Config()
    for key, default_value in config.to_dict().items():
        if key in ['optimizer_clip_value', 'optimizer_clip_norm']:
            default_value_type = float
        else:
            default_value_type = type(default_value)
        if default_value_type in [dict]:
            continue

        parser.add_argument('--{}'.format(key), type=default_value_type, default=default_value)
    args = parser.parse_args()

    config = Config(**vars(args))
    config.pretrained_config = parse_pretrained_config(config)
    return config


def compose_correlation_to_tokens(correlations, tokens, sentence, replacements):
    def _matches_candidates(index, matching_word):
        for original, replaced_tokens in replacements:
            query_tokens = tokens[index:min(index + len(replaced_tokens), len(tokens))]
            all_matched = all(a == b for a, b in zip(query_tokens, replaced_tokens))
            length = min(len(matching_word), len(original))
            if all_matched and matching_word.lower()[:length] == original.lower()[:length]:
                return True, (len(original), len(replaced_tokens))
        return False, (None, None)

    offset = 0
    skips = 0
    maxlen = len(sentence)
    composed_tokens = []
    for i, token in enumerate(tokens):
        if skips > 0:
            skips -= 1
            continue

        is_last_token = i == len(tokens) - 1
        while offset < len(sentence):
            matched = True
            if token.startswith('##'):
                matched = sentence[offset - 1] != ' '
                token = token[2:]

            word = sentence[offset:] if is_last_token else sentence[offset:min(offset + len(token), maxlen)]
            candidate_matched, (candidate_size, token_offset) = _matches_candidates(i, word)
            matched = matched and (token == word.lower() or candidate_matched)
            if candidate_matched:
                word = word[:candidate_size]
                skips += token_offset - 1
                matched_tokens = tokens[i:i + token_offset]
                matched_token_indices = list(range(i, i + token_offset))
            else:
                matched_tokens = [token]
                matched_token_indices = [i]

            if matched:
                score = correlations[i]
                composed_tokens.append({
                    'matched': True,
                    'text': word,
                    'tokens': matched_tokens,
                    'token_indices': matched_token_indices,
                    'score': float(score)
                })
                offset += len(word)
                break
            else:
                word = sentence[offset:] if is_last_token else sentence[offset]
                composed_tokens.append({
                    'matched': False,
                    'text': word,
                    'tokens': [],
                    'token_indices': [],
                    'score': 0.0
                })
                offset += len(word)
                if token in DISALLOWED_TOKENS:
                    break
    return composed_tokens
