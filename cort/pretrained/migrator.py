import os
import logging

import numpy as np
import tensorflow as tf

from transformers import TFElectraModel, ElectraConfig
from transformers import TFBertModel, BertConfig


def create_base_bert_config(vocab):
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
    config.pad_token_id = vocab['[PAD]']
    config.cls_token_id = vocab['[CLS]']
    config.sep_token_id = vocab['[SEP]']
    config.unk_token_id = vocab['[UNK]']
    return config


def create_base_electra_config(vocab):
    config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
    config.vocab_size = 16200
    config.pad_token_id = vocab['[PAD]']
    config.cls_token_id = vocab['[CLS]']
    config.sep_token_id = vocab['[SEP]']
    config.unk_token_id = vocab['[UNK]']
    return config


def create_base_electra(vocab, name=None):
    config = create_base_electra_config(vocab)
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


def create_base_bert(vocab, name=None):
    config = create_base_bert_config(vocab)
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


def migrate_electra(ckpt_dir_or_file, vocab, name=None):
    electra = create_base_electra(vocab, name=name)
    disallows = ['adam', 'generator', 'global_step', 'discriminator']
    return migrate_internal(electra, ckpt_dir_or_file, 'electra_mappings.txt', disallows)


def migrate_bert(ckpt_dir_or_file, vocab, name=None):
    bert = create_base_bert(vocab, name=name)
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


def restore_from_checkpoint(replica, ckpt_dir_or_file: str):
    assert replica.name == 'model', (
        'Pre-trained replica model name must be `model`'
    )
    replica(replica.dummy_inputs)  # evaluate to compile model graphs

    disallows = ['optimizer/', 'save_counter/', 'step/', '_checkpointable_object_graph']

    def _is_disallows(name):
        for disallow in disallows:
            if disallow in name:
                return True
        return False

    def _preprocess_ckpt_var_name(ckpt_var_name):
        orig = ckpt_var_name
        ckpt_var_name = ckpt_var_name.replace('/.ATTRIBUTES/VARIABLE_VALUE', '')
        ckpt_var_name = ckpt_var_name.replace('/attention/self_attention/', '/attention/self/')
        ckpt_var_name = ckpt_var_name.replace('/attention/dense_output/', '/attention/output/')
        ckpt_var_name = ckpt_var_name.replace('/bert_output/', '/output/')
        ckpt_var_name = ckpt_var_name.replace('/embeddings/weight', '/embeddings/word_embeddings/weight')
        ckpt_var_name = ckpt_var_name.replace('/embeddings/embeddings', '/embeddings/position_embeddings/embeddings')
        ckpt_var_name = ckpt_var_name.replace(
            '/embeddings/token_type_embeddings', '/embeddings/token_type_embeddings/embeddings'
        )
        return orig, ckpt_var_name

    reader = tf.train.load_checkpoint(ckpt_dir_or_file)
    shapes = reader.get_variable_to_shape_map()
    ckpt_vars = []
    for var_name, shape in shapes.items():
        uncased_var_name = var_name.lower()
        if _is_disallows(uncased_var_name):
            continue
        ckpt_vars.append((var_name, shape))
    ckpt_var_names = [_preprocess_ckpt_var_name(ckpt_var[0]) for ckpt_var in ckpt_vars]

    def _find_matching_variable(replica_var_name):
        replica_var_name = replica_var_name.split(':')[0] if ':' in replica_var_name else replica_var_name
        replica_var_name = replica_var_name.replace('layer_._', 'layer/')

        # exceptional condition for `projection` in CortForPretraining
        if 'model/' not in replica_var_name:
            replica_var_name = 'model/cort/' + replica_var_name

        for orig, ckpt_var_name in ckpt_var_names:
            if replica_var_name == ckpt_var_name:
                return orig
        return None

    invalid_var_names = []
    for variable in replica.variables:
        matched_ckpt_var_name = _find_matching_variable(variable.name)
        if matched_ckpt_var_name is None:
            invalid_var_names.append(variable.name)
            continue

        ckpt_var = tf.Variable(reader.get_tensor(matched_ckpt_var_name))
        if ckpt_var.shape != variable.shape:
            logging.warning('Variable shape mismatch: {}:{} <-> {}:{}'.format(
                matched_ckpt_var_name, ckpt_var.shape, variable.shape, variable.name
            ))
        variable.assign(ckpt_var)

    if len(invalid_var_names):
        logging.warning('Unresolved replica model variables: [{}]'.format(', '.join(invalid_var_names)))

    return replica
