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
from cort.modeling import CortModel, CortForElaboratedRepresentation
from cort.optimization import LinearWarmUp, AdamWeightDecay, GradientAccumulator
from cort.preprocessing import parse_and_preprocess_sentences, normalize_texts, run_multiprocessing_job
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


def preprocess_sentences_on_batch(batch):
    sentences = []
    for sentence in batch:
        sentence = normalize_texts(sentence)
        sentences.append(sentence)
    return sentences


def setup_datagen(config: Config):
    df = parse_and_preprocess_sentences(config.train_path)
    tokenizer = create_tokenizer_from_config(config)

    # preprocess in multiprocessing manner
    results = run_multiprocessing_job(preprocess_sentences_on_batch, df['sentences'],
                                      num_processes=config.num_processes)
    sentences = []
    for sentences_batch in results:
        sentences += sentences_batch

    tokenized = tokenizer(sentences,
                          padding='max_length',
                          truncation=True,
                          return_attention_mask=False,
                          return_token_type_ids=False)
    input_ids = tokenized['input_ids']
    sections = df['code_sections'].values
    labels = df['code_labels'].values

    input_ids = np.array(input_ids, dtype=np.int32)
    sections = np.array(sections, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    return input_ids, (sections, labels)


def splits_into_fold(config: Config, input_ids, labels):
    sections, labels = labels
    fold = StratifiedKFold(n_splits=config.num_k_fold, shuffle=True, random_state=config.seed)
    for index, (train_indices, valid_indices) in enumerate(fold.split(input_ids, labels)):
        if index != config.current_fold:
            continue

        train_input_ids = input_ids[train_indices]
        train_sections = sections[train_indices]
        train_labels = labels[train_indices]
        valid_input_ids = input_ids[valid_indices]
        valid_sections = sections[valid_indices]
        valid_labels = labels[valid_indices]

        steps_per_epoch = len(train_input_ids) // config.batch_size // config.gradient_accumulation_steps
        sections_cw = compute_class_weight('balanced', classes=np.unique(train_sections), y=train_sections)
        sections_cw = dict(enumerate(sections_cw))
        labels_cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        labels_cw = dict(enumerate(labels_cw))

        logging.info('Class weights:')
        logging.info('- Sections:')
        for i in range(config.num_sections):
            logging.info('  - Section #{}: {}'.format(i, sections_cw[i]))
        logging.info('- Labels:')
        for i in range(config.num_labels):
            logging.info('  - Label #{}: {}'.format(i, labels_cw[i]))

        FoldedDatasetOutput = collections.namedtuple('FoldedDatasetOutput', [
            'training', 'validation', 'steps_per_epoch',
            'sections_cw', 'labels_cw'
        ])
        return FoldedDatasetOutput(training=(train_input_ids, (train_sections, train_labels)),
                                   validation=(valid_input_ids, (valid_sections, valid_labels)),
                                   steps_per_epoch=steps_per_epoch,
                                   sections_cw=sections_cw, labels_cw=labels_cw)

    raise ValueError('Invalid current fold number: {} out of total {} folds'
                     .format(config.current_fold, config.num_k_fold))


def class_weight_map_fn(sections_cw, labels_cw):
    def _calc_cw_tensor(cw):
        class_ids = list(sorted(cw.keys()))
        expected_class_ids = list(range(len(class_ids)))
        if class_ids != expected_class_ids:
            raise ValueError((
                'Expected `class_weight` to be a dict with keys from 0 to one less'
                'than the number of classes, found {}'.format(cw)
            ))

        return tf.convert_to_tensor([cw[int(c)] for c in class_ids])

    sections_cw_tensor = _calc_cw_tensor(sections_cw)
    labels_cw_tensor = _calc_cw_tensor(labels_cw)

    @tf.function
    def _rearrange_cw(labels, cw_tensor):
        y_classes = smart_cond.smart_cond(
            labels.shape.rank == 2 and tf.shape(labels)[1] > 1,
            lambda: tf.argmax(labels, axis=1),
            lambda: tf.cast(tf.reshape(labels, (-1,)), dtype=tf.int32)
        )
        cw = tf.gather(cw_tensor, y_classes)
        return cw

    @tf.function
    def map_fn(input_ids, labels):
        sections, labels = labels

        sec_cw = _rearrange_cw(sections, sections_cw_tensor)
        cw = _rearrange_cw(labels, labels_cw_tensor)
        return input_ids, (sections, labels), (sec_cw, cw)

    return map_fn


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)


def run_train(strategy, config, train_dataset, valid_dataset, steps_per_epoch):
    # Build CoRT models
    if config.include_sections:
        model = CortForElaboratedRepresentation(config)
    else:
        model = CortModel(config)

    if config.restore_checkpoint:
        model.load_weights(config.restore_checkpoint)
    accumulator = GradientAccumulator()
    total_train_steps = config.epochs * steps_per_epoch
    warmup_apical_steps = int(max(1, total_train_steps * config.warmup_rate))
    num_steps = 0

    # Optimization
    def _get_layer_decay(decay_rate, num_layers):
        key_to_depths = collections.OrderedDict({
            '/embedding/': 0,
            '/embeddings/': 0,
            '/embeddings_project/': 0
        })
        total_depth = 0
        for layer in range(num_layers):
            total_depth += 1
            key_to_depths['/layer_._{}/'.format(layer)] = total_depth

        key_to_depths['/seq_repr/'] = total_depth + 1
        key_to_depths['/bi_seq_repr/'] = total_depth + 1

        # for elaborated representation model headings
        key_to_depths['/sec_repr/'] = total_depth + 1
        key_to_depths['/bi_sec_repr/'] = total_depth + 1

        return {
            key: decay_rate ** (total_depth + 1 - depth)
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
            decay_steps=total_train_steps - warmup_apical_steps,
            end_learning_rate=0.0,
            power=config.lr_poly_decay_power
        )
    elif config.lr_fn == 'linear_decay':
        learning_rate_fn = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=total_train_steps - warmup_apical_steps,
            end_learning_rate=0.0,
            power=1.0
        )
    elif config.lr_fn == 'constant':
        def constant_learning_rate(step):
            return config.learning_rate

        learning_rate_fn = constant_learning_rate
    else:
        raise ValueError('Invalid learning rate function type:', config.lr_fn)

    if config.warmup_rate:
        learning_rate_fn = LinearWarmUp(
            initial_learning_rate=config.learning_rate,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=warmup_apical_steps
        )

    layer_decay = None
    if config.layerwise_lr_decay:
        layer_decay = _get_layer_decay(config.layerwise_lr_decay, config.pretrained_config.num_hidden_layers)
        logging.info('Layer-wise LR Decay:')
        for pattern, rate in layer_decay.items():
            logging.info('- {}: {:.06f}'.format(pattern, rate))

    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=config.weight_decay,
        layer_decay=layer_decay,
        epsilon=1e-6,
        exclude_from_weight_decay=['layer_norm', 'bias', 'LayerNorm'],
        clip_norm=config.optimizer_clip_norm
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
        metric_map['micro_f1_score'] = metrics_tfa.F1Score(
            num_classes=config.num_labels, average='micro', name='micro_f1_score'
        )
        metric_map['macro_f1_score'] = metrics_tfa.F1Score(
            num_classes=config.num_labels, average='macro', name='macro_f1_score'
        )

        # metrics for CoRT with elaborated representation
        if config.include_sections:
            metric_map['section_contrastive_loss'] = metrics.Mean(name='section_contrastive_loss')
            metric_map['section_cross_entropy_loss'] = metrics.Mean(name='section_cross_entropy_loss')
            metric_map['section_accuracy'] = metrics.CategoricalAccuracy(name='section_accuracy')
            metric_map['section_precision'] = metrics.Precision(name='section_precision')
            metric_map['section_recall'] = metrics.Recall(name='section_recall')
            metric_map['section_micro_f1_score'] = metrics_tfa.F1Score(
                num_classes=config.num_sections, average='micro', name='section_micro_f1_score'
            )
            metric_map['section_macro_f1_score'] = metrics_tfa.F1Score(
                num_classes=config.num_sections, average='macro', name='section_macro_f1_score'
            )
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
        if config.include_sections:
            # metrics for CoRT with elaborated representation
            dicts['section_contrastive_loss'].update_state(d['section_contrastive_loss'])
            dicts['section_cross_entropy_loss'].update_state(d['section_cross_entropy_loss'])
            confusion_keys = ['section_' + key for key in confusion_keys]
            for key in confusion_keys:
                dicts[key].update_state(
                    y_true=d['section_ohe_labels'],
                    y_pred=d['section_probs']
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

    ckpt_file_name = 'CoRT-SWEEP_{}-RUN_{}.h5'.format(wandb.run.sweep_id, wandb.run.id)
    # Callbacks
    callback_list = [
        callbacks.ModelCheckpoint(os.path.join('./models', ckpt_file_name),
                                  monitor='val_total_loss',
                                  verbose=1, save_best_only=True, save_weights_only=True),
        wandb.keras.WandbCallback()
    ]
    callback = callbacks.CallbackList(callbacks=callback_list,
                                      model=model,
                                      epochs=config.epochs,
                                      steps=steps_per_epoch)

    # Training
    def strategy_reduce_mean(unscaled_loss, outputs):
        features = dict()
        for key, value in outputs.items():
            features[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
        unscaled_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, unscaled_loss, axis=None)
        return unscaled_loss, features

    def wrap_inputs(inputs):
        if config.include_sections:
            return inputs

        input_ids, (_, labels), (_, cw) = inputs
        return input_ids, labels, cw

    @tf.function
    def train_step(inputs, take_step):
        with tf.GradientTape() as tape:
            total_loss, outputs = model(wrap_inputs(inputs), training=True)
            unscaled_loss = tf.stop_gradient(total_loss)
        grads = tape.gradient(total_loss, model.trainable_variables)

        # Accumulate gradients
        accumulator(grads)
        if take_step:
            # All reduce and clip the accumulated gradients
            reduced_accumulated_gradients = [
                None if g is None else g / tf.cast(config.gradient_accumulation_steps, g.dtype)
                for g in accumulator.accumulated_gradients
            ]
            (clipped_accumulated_gradients, _) = tf.clip_by_global_norm(reduced_accumulated_gradients, clip_norm=1.0)

            # Weight update
            optimizer.apply_gradients(zip(clipped_accumulated_gradients, model.trainable_variables))
            accumulator.reset()

        return unscaled_loss, outputs

    @tf.function
    def distributed_train_step(inputs, take_step):
        unscaled_loss, outputs = strategy.run(train_step, args=(inputs, take_step))
        return strategy_reduce_mean(unscaled_loss, outputs)

    @tf.function
    def test_step(inputs):
        return model(wrap_inputs(inputs), training=False)

    @tf.function
    def distributed_test_step(inputs,):
        unscaled_loss, outputs = strategy.run(test_step, args=(inputs,))
        return strategy_reduce_mean(unscaled_loss, outputs)

    train_fn = distributed_train_step if config.distribute else train_step
    test_fn = distributed_test_step if config.distribute else test_step

    def evaluate(run_callback=True):
        if run_callback:
            callback.on_test_begin()
        for index, inputs in enumerate(valid_dataset):
            if run_callback:
                callback.on_test_batch_begin(index)
            total_loss, outputs = test_fn(inputs)

            # assign new metrics
            metric_maps['valid']['total_loss'].update_state(values=total_loss)
            metric_fn(metric_maps['valid'], outputs)

            if run_callback:
                callback.on_test_batch_end(index, logs=create_metric_logs(metric_maps['valid']))
        logs = create_metric_logs(metric_maps['valid'])
        if run_callback:
            callback.on_test_end(logs)

        val_logs = {'val_' + key: value for key, value in logs.items()}
        # WandB step-wise logging after evaluation
        wandb.log(val_logs, step=num_steps)
        return val_logs

    def reset_metrics():
        # reset all metric states
        for key in metric_maps.keys():
            [metric.reset_state() for metric in metric_maps[key].values()]

    if not config.skip_early_eval:
        # very first evaluate for initial metric results
        evaluate(run_callback=False)
    else:
        logging.info('Skipping early evaluation')

    training_logs = None
    callback.on_train_begin()
    for epoch in range(config.initial_epoch, config.epochs):
        reset_metrics()
        callback.on_epoch_begin(epoch)
        print('\nEpoch {}/{}'.format(epoch + 1, config.epochs))

        progbar = utils.Progbar(steps_per_epoch,
                                stateful_metrics=[metric.name for metric in metric_maps['train'].values()])
        accumulator.reset()
        local_step = 0
        for step, input_batches in enumerate(train_dataset.take(steps_per_epoch * config.gradient_accumulation_steps)):
            # Need to call apply_gradients on very first step irrespective of gradient accumulation
            # This is required for the optimizer to build its states
            accumulation_step = (step + 1) % config.gradient_accumulation_steps == 0 or num_steps == 0
            if accumulation_step:
                callback.on_train_batch_begin(local_step)

            training_loss, eval_inputs = train_fn(
                input_batches, take_step=accumulation_step
            )

            if accumulation_step:
                # assign new metrics
                metric_maps['train']['total_loss'].update_state(values=training_loss)
                metric_fn(metric_maps['train'], eval_inputs)

                batch_logs = create_metric_logs(metric_maps['train'])
                progbar.update(local_step, values=[(k, v) for k, v in batch_logs.items()])

                # WandB step-wise logging during training
                wandb.log(batch_logs, step=num_steps)
                wandb.log({
                    'learning_rate': learning_rate_fn(num_steps)
                }, step=num_steps)

                callback.on_train_batch_end(local_step, logs=batch_logs)
                local_step += 1
                num_steps += 1

        train_logs = create_metric_logs(metric_maps['train'])
        epoch_logs = copy.copy(train_logs)

        eval_logs = evaluate()
        epoch_logs.update(eval_logs)

        progbar.update(steps_per_epoch,
                       values=[(k, v) for k, v in epoch_logs.items()],
                       finalize=True)
        training_logs = epoch_logs
        callback.on_epoch_end(epoch, logs=epoch_logs)
    callback.on_train_end(training_logs)


def main():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(Formatter())

    logging.basicConfig(level=logging.DEBUG, handlers=[
        handler
    ])

    config = parse_arguments()

    if config.cross_validation == 'kfold':
        wandb.init(project='CoRT', name='CoRT-KFOLD_{}'.format(config.current_fold + 1))
    elif config.cross_validation == 'hyperparams':
        wandb.init(project='CoRT')
    else:
        raise ValueError('Invalid CV strategy: {}'.format(config.cross_validation))

    logging.info('WandB setup:')
    logging.info('- Sweep ID: {}'.format(wandb.run.sweep_id))
    logging.info('- Run ID: {}'.format(wandb.run.id))

    with tf.device('/device:GPU:{}'.format(config.gpu)) if config.gpu != 'all' else empty_context_manager():
        strategy = tf.distribute.MirroredStrategy()
        if config.distribute:
            logging.info('Distributed Training Enabled')
            config.batch_size = config.batch_size * strategy.num_replicas_in_sync

        if config.include_sections:
            logging.info('Elaborated Representation Enabled')

        if config.repr_preact:
            logging.info('Pre-Activated Representation Enabled')

        set_random_seed(config.seed)

        input_ids, labels = setup_datagen(config)
        folded_output = splits_into_fold(config, input_ids, labels)

        train_dataset = tf.data.Dataset.from_tensor_slices(folded_output.training)
        train_dataset = train_dataset.map(class_weight_map_fn(folded_output.sections_cw, folded_output.labels_cw))
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
