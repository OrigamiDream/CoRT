import os
import wandb
import random
import argparse

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from cort.config import Config
from cort.modeling import CortModel
from cort.preprocessing import parse_and_preprocess_sentences
# from pretrained.tokenization import create_tokenizer
# from pretrained.config import create_bert_config, create_electra_config
from transformers import AutoTokenizer, AutoConfig
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import callbacks, optimizers, metrics
from tensorflow_addons import metrics as metrics_tfa, callbacks as callbacks_tfa
from sklearn.model_selection import StratifiedKFold


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
    # if config.model_name == 'korscibert':
    #     config.pretrained_config = create_bert_config()
    # elif config.model_name == 'korscielectra':
    #     config.pretrained_config = create_electra_config()
    # else:
    #     config.pretrained_config = AutoConfig.from_pretrained(config.model_name)
    config.pretrained_config = AutoConfig.from_pretrained(config.model_name)

    config.pretrained_config.max_position_embeddings = min(512, config.pretrained_config.max_position_embeddings)
    return config


def setup_datagen(config: Config):
    df = parse_and_preprocess_sentences(config.train_path, debug=True)

    # if config.model_name == 'korscibert':
    #     tokenizer = create_tokenizer('./pretrained/korscibert/vocab_kisti.txt', tokenizer_type='bert')
    # elif config.model_name == 'korscielectra':
    #     tokenizer = create_tokenizer('./pretrained/korscielectra/data/vocab.txt', tokenizer_type='electra')
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # pretrained_config = config.pretrained_config
    # max_length = pretrained_config.max_position_embeddings
    #
    # cls_token = tokenizer.convert_tokens_to_ids('[CLS]')
    # sep_token = tokenizer.convert_tokens_to_ids('[SEP]')
    # pad_token = tokenizer.convert_tokens_to_ids('[PAD]')
    # num_reserved_tokens = 2  # 2 reserved tokens for [CLS] and [SEP]
    #
    # input_ids = []
    # labels = []
    # for sentence, label in tqdm(zip(df['sentences'], df['code_labels']), total=len(df['sentences'])):
    #     tokens = tokenizer.tokenize(sentence)[:max_length - num_reserved_tokens]
    #     tokens = tokenizer.convert_tokens_to_ids(tokens)
    #
    #     tokens = [cls_token] + tokens + [sep_token]
    #
    #     num_pads = max_length - len(tokens)  # add paddings
    #     tokens = tokens + [pad_token] * num_pads
    #
    #     input_ids.append(tokens)
    #     labels.append(label)
    #
    # input_ids = sequence.pad_sequences(input_ids,
    #                                    maxlen=max_length,
    #                                    padding='post', truncating='post',
    #                                    value=pad_token)

    tokenized = tokenizer(list(df['sentences']),
                          padding='max_length',
                          truncation=True,
                          return_attention_mask=False,
                          return_token_type_ids=False)
    input_ids = tokenized['input_ids']
    labels = df['code_labels'].values

    input_ids = np.array(input_ids, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    return input_ids, labels


def splits_into_fold(config: Config, input_ids, labels):
    fold = StratifiedKFold(n_splits=config.num_k_fold, shuffle=True, random_state=config.seed)
    for index, (train_indices, valid_indices) in enumerate(fold.split(input_ids, labels)):
        if index != config.current_fold:
            continue

        train_input_ids = input_ids[train_indices]
        train_labels = labels[train_indices]
        valid_input_ids = input_ids[valid_indices]
        valid_labels = labels[valid_indices]

        steps_per_epoch = len(train_input_ids) // config.batch_size
        return (train_input_ids, train_labels), (valid_input_ids, valid_labels), steps_per_epoch

    raise ValueError('Invalid current fold number: {} out of total {} folds'
                     .format(config.current_fold, config.num_k_fold))


def class_weight_map_fn(class_weight):
    class_ids = list(sorted(class_weight.keys()))
    expected_class_ids = list(range(len(class_ids)))
    if class_ids != expected_class_ids:
        raise ValueError((
            'Expected `class_weight` to be a dict with keys from 0 to one less'
            'than the number of classes, found {}'.format(class_weight)
        ))

    class_weight_tensor = tf.convert_to_tensor([class_weight[int(c)] for c in class_ids])

    def map_fn(input_ids, labels):
        y_classes = tf.cond(
            labels.shape.rank == 2 and tf.shape(labels)[1] > 1,
            lambda: tf.argmax(labels, axis=1),
            lambda: tf.cast(tf.reshape(labels, (-1,)), dtype=tf.int64)
        )
        cw = tf.gather(class_weight_tensor, y_classes)
        return input_ids, labels, cw

    return map_fn


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)


def run_train(config, train_dataset, valid_dataset, steps_per_epoch):
    model = CortModel(config)
    optimizer = optimizers.Adam(learning_rate=config.learning_rate)

    def create_metric_map():
        metric_map = dict()
        metric_map['total_loss'] = metrics.Mean(name='total_loss')
        metric_map['contrastive_loss'] = metrics.Mean(name='contrastive_loss')
        metric_map['cross_entropy_loss'] = metrics.Mean(name='cross_entropy_loss')
        metric_map['accuracy'] = metrics.Accuracy(name='accuracy')
        metric_map['precision'] = metrics.Precision(name='precision')
        metric_map['recall'] = metrics.Recall(name='recall')
        metric_map['micro_f1_score'] = metrics_tfa.F1Score(num_classes=config.num_labels, average='micro')
        metric_map['macro_f1_score'] = metrics_tfa.F1Score(num_classes=config.num_labels, average='macro')
        return metric_map

    def metric_fn(dicts, model_outputs):
        d = model_outputs
        dicts['contrastive_loss'].update_state(d['contrastive_loss'])
        dicts['cross_entropy_loss'].update_state(d['cross_entropy_loss'])
        dicts['accuracy'].update_state(
            y_true=d['ohe_labels'],
            y_pred=d['probs']
        )
        dicts['precision'].update_state(
            y_true=d['ohe_labels'],
            y_pred=d['probs']
        )
        dicts['recall'].update_state(
            y_true=d['ohe_labels'],
            y_pred=d['probs']
        )
        dicts['micro_f1_score'].update_state(
            y_true=d['ohe_labels'],
            y_pred=d['probs']
        )
        dicts['macro_f1_score'].update_state(
            y_true=d['ohe_labels'],
            y_pred=d['probs']
        )
        return dicts

    def create_metric_logs(dicts):
        metric_logs = {}
        for k, v in dicts.items():
            value = float(v.result().numpy())
            metric_logs[k] = value
        return metric_logs

    metric_maps = {
        'train': create_metric_map(),
        'valid': create_metric_map()
    }

    callback = callbacks.CallbackList(callbacks=[
        callbacks_tfa.TQDMProgressBar(update_per_second=1),
        wandb.keras.WandbCallback()
    ])
    callback.set_model(model)
    callback.set_params({
        'epochs': config.epochs,
        'steps': steps_per_epoch,
    })

    @tf.function
    def train_step(input_ids, labels):
        with tf.GradientTape() as tape:
            total_loss, outputs = model([input_ids, labels], training=True)
            unscaled_loss = tf.stop_gradient(total_loss)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return unscaled_loss, outputs

    @tf.function
    def test_step(input_ids, labels):
        return model([input_ids, labels], training=False)

    def train_one_step():
        for index, (input_ids, labels) in enumerate(train_dataset):
            callback.on_train_batch_begin(index)
            total_loss, outputs = train_step(input_ids, labels)

            # assign new metrics
            metric_maps['train']['total_loss'].update_state(values=total_loss)
            metric_fn(metric_maps['train'], outputs)

            callback.on_train_batch_end(index, logs=create_metric_logs(metric_maps['train']))
        return create_metric_logs(metric_maps['train'])

    def test_one_step():
        for index, (input_ids, labels) in enumerate(valid_dataset):
            callback.on_test_batch_begin(index)
            total_loss, outputs = test_step(input_ids, labels)

            # assign new metrics
            metric_maps['valid']['total_loss'].update_state(values=total_loss)
            metric_fn(metric_maps['valid'], outputs)

            callback.on_test_batch_begin(index, logs=create_metric_logs(metric_maps['valid']))
        return create_metric_logs(metric_maps['valid'])

    # current_step = 1
    for epoch in range(config.initial_epoch, config.epochs):
        callback.on_epoch_begin(epoch)
        logs = {}

        train_logs = train_one_step()
        logs.update(train_logs)

        test_logs = test_one_step()
        test_logs = {'val_' + key: value for key, value in test_logs.items()}
        logs.update(test_logs)

        wandb.log(logs, step=epoch + 1)

        # reset all metric states
        for key in metric_maps.keys():
            [metric.reset_state() for metric in metric_maps[key].values()]

        callback.on_epoch_end(epoch, logs=logs)


def main():
    config = parse_arguments()
    wandb.init(project='CoRT', name='CoRT-FOLD_{}'.format(config.current_fold + 1))

    set_random_seed(config.seed)

    input_ids, labels = setup_datagen(config)
    train_data, valid_data, steps_per_epoch = splits_into_fold(config, input_ids, labels)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(config.batch_size).shuffle(buffer_size=1024).repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
    valid_dataset = valid_dataset.batch(config.batch_size)

    run_train(config, train_dataset, valid_dataset, steps_per_epoch)


if __name__ == '__main__':
    main()
