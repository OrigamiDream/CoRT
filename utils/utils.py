import os
import time
import random
import argparse
import contextlib
import numpy as np
import tensorflow as tf

from cort.config import Config
from cort.pretrained import migrator, tokenization
from transformers import AutoConfig, AutoTokenizer


@contextlib.contextmanager
def empty_context_manager():
    yield None


def generate_random_id(length=8):
    chars = 'abcdefghijklnmopqrstuvwxyz0123456789'
    return ''.join([random.choice(chars) for i in range(length)])


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
    if config.model_name == 'korscibert':
        vocab = parse_vocabulary(config.korscibert_vocab)
        config.pretrained_config = migrator.create_base_bert_config(pad_token_id=vocab['[PAD]'])
    elif config.model_name == 'korscielectra':
        vocab = parse_vocabulary(config.korscielectra_vocab)
        config.pretrained_config = migrator.create_base_electra_config(pad_token_id=vocab['[PAD]'])
    else:
        config.pretrained_config = AutoConfig.from_pretrained(config.model_name)

    config.pretrained_config.max_position_embeddings = min(512, config.pretrained_config.max_position_embeddings)
    return config
