import os
import json

from pretrained.korscibert.modeling import BertConfig
from pretrained.korscielectra.model.modeling import BertConfig as ElectraConfig


def current_directory_path() -> str:
    return os.path.dirname(__file__)


def create_bert_config() -> BertConfig:
    path = os.path.join(current_directory_path(), 'korscibert', 'bert_config_kisti.json')
    with open(path, 'r') as f:
        config = json.load(f)

    return BertConfig(vocab_size=config['vocab_size'],
                      hidden_size=config['hidden_size'],
                      num_hidden_layers=config['num_hidden_layers'],
                      num_attention_heads=config['num_attention_heads'],
                      intermediate_size=config['intermediate_size'],
                      hidden_act=config['hidden_act'],
                      hidden_dropout_prob=config['hidden_dropout_prob'],
                      attention_probs_dropout_prob=config['attention_probs_dropout_prob'],
                      max_position_embeddings=config['max_position_embeddings'],
                      type_vocab_size=config['type_vocab_size'],
                      initializer_range=config['initializer_range'])


def create_electra_config() -> ElectraConfig:
    path = os.path.join(current_directory_path(), 'korscielectra', 'korsci_config.json')
    with open(path, 'r') as f:
        config = json.load(f)

    return BertConfig(vocab_size=config['vocab_size'],
                      max_position_embeddings=config['max_seq_length'])
