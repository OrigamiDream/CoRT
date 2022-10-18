import os

import numpy as np
import tensorflow as tf

from transformers import TFElectraModel, ElectraConfig
from transformers import TFBertModel, BertConfig


def create_base_bert_config(pad_token_id):
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.attention_probs_dropout_prob = 0.1
    config.hidden_act = 'gelu'
    config.hidden_dropout_prob = 0.1
    config.hidden_size = 768
    config.initializer_range = 0.02
    config.intermediate_size = 3072
    config.max_position_embeddings = 512
    config.num_attention_heads = 12
    config.num_hidden_layers = 12
    config.type_vocab_size = 2
    config.vocab_size = 15330
    config.pad_token_id = pad_token_id
    return config


def create_base_electra_config(pad_token_id):
    config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
    config.vocab_size = 16200
    config.pad_token_id = pad_token_id
    return config


def create_base_electra(pad_token_id, name=None):
    config = create_base_electra_config(pad_token_id)
    batch_size = 1
    eval_shape = (batch_size, config.max_position_embeddings)
    eval_inputs = {
        'input_ids': np.zeros(eval_shape, dtype=np.int32),
        'attention_mask': np.zeros(eval_shape, dtype=np.int32),
        'token_type_ids': np.zeros(eval_shape, dtype=np.int32)
    }
    electra = TFElectraModel(config, name=name)
    electra(**eval_inputs)  # callable
    return electra


def create_base_bert(pad_token_id, name=None):
    config = create_base_bert_config(pad_token_id)
    batch_size = 1
    eval_shape = (batch_size, config.max_position_embeddings)
    eval_inputs = {
        'input_ids': np.zeros(eval_shape, dtype=np.int32),
        'attention_mask': np.zeros(eval_shape, dtype=np.int32),
        'token_type_ids': np.zeros(eval_shape, dtype=np.int32)
    }
    bert = TFBertModel(config, name=name)
    bert(**eval_inputs)  # callable
    return bert


def read_var_mappings(mapping_name):
    path = os.path.join(os.path.dirname(__file__), 'mappings', mapping_name)  # relative path
    with open(path, 'r') as f:
        texts = f.read().splitlines()
        mappings = {}
        for text in texts:
            splits = text.split(',')
            assert len(splits) == 2, (
                'Number of mapping splits must be 2, but received {} instead'.format(len(splits))
            )
            mappings[splits[0]] = splits[1]
    return mappings


def migrate_electra(ckpt_dir_or_file, pad_token_id, name=None):
    electra = create_base_electra(pad_token_id, name=name)
    disallows = ['adam', 'generator', 'global_step', 'discriminator']
    return migrate_internal(electra, ckpt_dir_or_file, 'electra_mappings.txt', disallows)


def migrate_bert(ckpt_dir_or_file, pad_token_id, name=None):
    bert = create_base_bert(pad_token_id, name=name)
    disallows = ['adam', 'global_step', 'good_steps', 'current_loss_scale', 'cls/']
    return migrate_internal(bert, ckpt_dir_or_file, 'bert_mappings.txt', disallows)


def migrate_internal(model, ckpt_dir_or_file, mapping_name, disallows):

    def _find_variable(name):
        for variable in model.variables:
            if variable.name == name:
                return variable
        raise ValueError('Failed to find variable by name: {}'.format(name))

    def _is_disallows(name):
        for disallow in disallows:
            if disallow in name:
                return True
        return False

    mappings = read_var_mappings(mapping_name)
    reader = tf.train.load_checkpoint(ckpt_dir_or_file)
    dtypes = reader.get_variable_to_dtype_map()
    unassigned_variable_names = [variable.name for variable in model.variables]
    for key in dtypes.keys():
        if _is_disallows(key):
            continue

        old_var = tf.Variable(reader.get_tensor(key))
        new_var_name = mappings[key].format(model_name=model.name)
        new_var = _find_variable(new_var_name)
        new_var.assign(old_var)
        unassigned_variable_names.remove(new_var_name)

    if len(unassigned_variable_names) > 0:
        print('Following variables are not initialized: [{}]'.format(', '.join(unassigned_variable_names)))

    return model
