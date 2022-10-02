import copy
import os
import wandb
import random
import logging
import argparse
import contextlib
import collections
import numpy as np
import tensorflow as tf

from cort.config import Config
from cort.modeling import CortModel
from cort.optimization import LinearWarmUp, AdamWeightDecay
from cort.preprocessing import parse_and_preprocess_sentences
from cort.pretrained import migrator, tokenization
from transformers import AutoTokenizer, AutoConfig
from tensorflow.keras import callbacks, optimizers, metrics, utils
from tensorflow.python.framework import smart_cond
from tensorflow_addons import metrics as metrics_tfa
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


class Formatter(logging.Formatter):

    default = '\x1b[38;5;250m'
    info = '\x1b[38;5;255m'
    debugging = '\x1b[38;5;245m'
    warning = '\x1b[33;20m'
    fatal = '\x1b[31;20m'
    resetting = '\x1b[0m'

    def format(self, record: logging.LogRecord) -> str:
        formats = self.default + '%(asctime)s - %(levelname)s: '
        if record.levelno == logging.DEBUG:
            formats += self.debugging
        elif record.levelno == logging.WARN:
            formats += self.warning
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            formats += self.fatal
        else:
            formats += self.info
        formats += '%(message)s' + self.resetting
        default_formatter = logging.Formatter(formats, datefmt='%H:%M:%S')
        return default_formatter.format(record)


@contextlib.contextmanager
def empty_context_manager():
    yield None


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
    tokenizer = create_tokenizer_from_config(config)
    if config.model_name == 'korscibert':
        config.pretrained_config = migrator.create_base_bert_config(tokenizer=tokenizer)
    elif config.model_name == 'korscielectra':
        config.pretrained_config = migrator.create_base_electra_config(tokenizer=tokenizer)
    else:
        config.pretrained_config = AutoConfig.from_pretrained(config.model_name)

    config.pretrained_config.max_position_embeddings = min(512, config.pretrained_config.max_position_embeddings)
    return config


def create_tokenizer_from_config(config):
    if config.model_name == 'korscibert':
        tokenizer = tokenization.create_tokenizer('./cort/pretrained/korscibert/vocab_kisti.txt',
                                                  tokenizer_type='bert')
    elif config.model_name == 'korscielectra':
        tokenizer = tokenization.create_tokenizer('./cort/pretrained/korscielectra/data/vocab.txt',
                                                  tokenizer_type='electra')
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer


def setup_datagen(config: Config):
    df = parse_and_preprocess_sentences(config.train_path)
    tokenizer = create_tokenizer_from_config(config)

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
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(train_labels),
                                             y=train_labels)
        class_weights = dict(enumerate(class_weights))
        logging.debug('Class weights:')
        for i in range(config.num_labels):
            logging.debug('- Label #{}: {}'.format(i, class_weights[i]))
        FoldedDatasetOutput = collections.namedtuple('FoldedDatasetOutput', [
            'training', 'validation',
            'steps_per_epoch', 'class_weights'
        ])
        return FoldedDatasetOutput(training=(train_input_ids, train_labels),
                                   validation=(valid_input_ids, valid_labels),
                                   steps_per_epoch=steps_per_epoch,
                                   class_weights=class_weights)

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
        y_classes = smart_cond.smart_cond(
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


def run_train(strategy, config, train_dataset, valid_dataset, steps_per_epoch):
    model = CortModel(config)

    # Optimization
    def _get_layer_decay(decay_rate, num_layers):
        key_to_depths = collections.OrderedDict({
            '/embedding/': 0,
            '/embeddings_project/': 0,
        })
        total_depth = 0
        for layer in range(num_layers):
            total_depth += 1
            key_to_depths['layer_._{}'.format(layer + 1)] = total_depth

        if 'bert' in config.model_name:
            total_depth += 1
            key_to_depths['/pooler/'] = total_depth

        total_depth += 1
        key_to_depths['/repr/'] = total_depth
        total_depth += 1
        key_to_depths['/classifier/'] = total_depth
        return {
            key: decay_rate ** (total_depth - depth)
            for key, depth in key_to_depths.items()
        }

    if config.lr_fn == 'cosine_decay':
        learning_rate_fn = optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=config.learning_rate,
            first_decay_steps=config.cosine_annealing_freq,
            t_mul=1.0, m_mul=1.0
        )
    elif config.lr_fn == 'polynomial_decay':
        learning_rate_fn = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=config.epochs - config.warmup_apical_epochs,
            end_learning_rate=0.0,
            power=config.lr_poly_decay_power
        )
    elif config.lr_fn == 'linear_decay':
        learning_rate_fn = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=config.epochs - config.warmup_apical_epochs,
            end_learning_rate=0.0,
            power=1.0
        )
    else:
        raise ValueError('Invalid learning rate function type:', config.lr_fn)

    if config.warmup_apical_epochs:
        learning_rate_fn = LinearWarmUp(
            initial_learning_rate=config.learning_rate,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=config.warmup_apical_epochs
        )

    layer_decay = None
    if config.layerwise_lr_decay:
        layer_decay = _get_layer_decay(config.layerwise_lr_decay, config.pretrained_config.num_hidden_layers)

    if config.weight_decay > 0.0:
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=config.weight_decay,
            layer_decay=layer_decay,
            exclude_from_weight_decay=['layer_norm', 'bias', 'LayerNorm'],
            clip_norm=config.optimizer_clip_norm
        )
    else:
        optimizer = optimizers.Adam(
            learning_rate=learning_rate_fn,
            clipnorm=config.optimizer_clip_norm
        )

    # Metrics
    def create_metric_map():
        metric_map = dict()
        metric_map['total_loss'] = metrics.Mean(name='total_loss')
        metric_map['contrastive_loss'] = metrics.Mean(name='contrastive_loss')
        metric_map['cross_entropy_loss'] = metrics.Mean(name='cross_entropy_loss')
        metric_map['accuracy'] = metrics.CategoricalAccuracy(name='accuracy')
        metric_map['precision'] = metrics.Precision(name='precision')
        metric_map['recall'] = metrics.Recall(name='recall')
        metric_map['micro_f1_score'] = metrics_tfa.F1Score(num_classes=config.num_labels, average='micro')
        metric_map['macro_f1_score'] = metrics_tfa.F1Score(num_classes=config.num_labels, average='macro')
        return metric_map

    def metric_fn(dicts, model_outputs):
        d = model_outputs
        dicts['contrastive_loss'].update_state(d['contrastive_loss'])
        dicts['cross_entropy_loss'].update_state(d['cross_entropy_loss'])
        confusion_keys = ['accuracy', 'precision', 'recall',
                          'micro_f1_score', 'macro_f1_score']
        for key in confusion_keys:
            dicts[key].update_state(
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

    # Callbacks
    callback = callbacks.CallbackList(callbacks=[
        callbacks.ModelCheckpoint('./models/CoRT-FOLD_{}-SWEEP_{}.h5'.format(config.current_fold, wandb.run.sweep_id),
                                  monitor='val_total_loss',
                                  verbose=1, save_best_only=True, save_weights_only=True),
        wandb.keras.WandbCallback()
    ])
    callback.set_model(model)
    callback.set_params({
        'epochs': config.epochs,
        'steps': steps_per_epoch,
    })

    # Training
    def strategy_reduce_mean(unscaled_loss, outputs):
        features = dict()
        for key, value in outputs.items():
            features[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
        unscaled_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, unscaled_loss, axis=None)
        return unscaled_loss, features

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            total_loss, outputs = model(list(inputs), training=True)
            unscaled_loss = tf.stop_gradient(total_loss)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return unscaled_loss, outputs

    @tf.function
    def distributed_train_step(inputs):
        unscaled_loss, outputs = strategy.run(train_step, args=(inputs,))
        return strategy_reduce_mean(unscaled_loss, outputs)

    @tf.function
    def test_step(inputs):
        return model(list(inputs), training=False)

    @tf.function
    def distributed_test_step(inputs):
        unscaled_loss, outputs = strategy.run(test_step, args=(inputs,))
        return strategy_reduce_mean(unscaled_loss, outputs)

    train_fn = distributed_train_step if config.distribute else train_step
    test_fn = distributed_test_step if config.distribute else test_step

    def train_one_step(progbar: utils.Progbar):
        for index, inputs in enumerate(train_dataset.take(steps_per_epoch)):
            callback.on_train_batch_begin(index)
            total_loss, outputs = train_fn(inputs)

            # assign new metrics
            metric_maps['train']['total_loss'].update_state(values=total_loss)
            metric_fn(metric_maps['train'], outputs)

            logs = create_metric_logs(metric_maps['train'])
            progbar.update(index, values=[(k, v) for k, v in logs.items()])

            callback.on_train_batch_end(index, logs=logs)
        return create_metric_logs(metric_maps['train'])

    def evaluate():
        callback.on_test_begin()
        for index, inputs in enumerate(valid_dataset):
            callback.on_test_batch_begin(index)
            total_loss, outputs = test_fn(inputs)

            # assign new metrics
            metric_maps['valid']['total_loss'].update_state(values=total_loss)
            metric_fn(metric_maps['valid'], outputs)

            callback.on_test_batch_end(index, logs=create_metric_logs(metric_maps['valid']))
        logs = create_metric_logs(metric_maps['valid'])
        callback.on_test_end(logs)
        return logs

    def reset_metrics():
        # reset all metric states
        for key in metric_maps.keys():
            [metric.reset_state() for metric in metric_maps[key].values()]

    training_logs = None
    callback.on_train_begin()
    for epoch in range(config.initial_epoch, config.epochs):
        reset_metrics()
        callback.on_epoch_begin(epoch)
        print('\nEpoch {}/{}'.format(epoch + 1, config.epochs))

        bar = utils.Progbar(steps_per_epoch,
                            stateful_metrics=[metric.name for metric in metric_maps['train'].values()])
        train_logs = train_one_step(bar)
        epoch_logs = copy.copy(train_logs)

        val_logs = evaluate()
        val_logs = {'val_' + key: value for key, value in val_logs.items()}
        epoch_logs.update(val_logs)

        bar.update(steps_per_epoch,
                   values=[(k, v) for k, v in epoch_logs.items()],
                   finalize=True)
        wandb.log(epoch_logs, step=epoch + 1)
        wandb.log({
            'learning_rate': learning_rate_fn(epoch)
        }, step=epoch + 1)
        training_logs = epoch_logs

        callback.on_epoch_end(epoch, logs=epoch_logs)
    callback.on_train_end(training_logs)


def main():
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(Formatter())

    logging.basicConfig(level=logging.DEBUG, handlers=[
        handler
    ])

    config = parse_arguments()
    wandb.init(project='CoRT', name='CoRT-FOLD_{}'.format(config.current_fold + 1))

    strategy = tf.distribute.MirroredStrategy()
    if config.distribute:
        logging.info('Distributed Training Enabled')
        config.batch_size = config.batch_size * strategy.num_replicas_in_sync

    set_random_seed(config.seed)

    input_ids, labels = setup_datagen(config)
    folded_output = splits_into_fold(config, input_ids, labels)

    train_dataset = tf.data.Dataset.from_tensor_slices(folded_output.training)
    train_dataset = train_dataset.map(class_weight_map_fn(folded_output.class_weights))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(config.batch_size).shuffle(buffer_size=1024).repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices(folded_output.validation)
    valid_dataset = valid_dataset.batch(config.batch_size)

    if config.distribute:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_dataset = strategy.experimental_distribute_dataset(train_dataset.with_options(options))
        valid_dataset = strategy.experimental_distribute_dataset(valid_dataset.with_options(options))

    with strategy.scope() if config.distribute else empty_context_manager():
        run_train(strategy, config, train_dataset, valid_dataset, folded_output.steps_per_epoch)


if __name__ == '__main__':
    main()
