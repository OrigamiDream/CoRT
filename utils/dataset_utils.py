import logging
import numpy as np
import tensorflow as tf

from cort.config import Config
from tensorflow.python.framework import smart_cond
from sklearn.utils.class_weight import compute_class_weight


def parse_tfrecords(config: Config):
    maxlen = config.pretrained_config.max_position_embeddings
    feature_desc = {
        'input_ids': tf.io.FixedLenFeature([maxlen], tf.int64),
        'sections': tf.io.FixedLenFeature([1], tf.int64),
        'labels': tf.io.FixedLenFeature([1], tf.int64)
    }

    def _parse_feature_desc(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_desc)

        # tf.int64 is acceptable, but tf.int32 has more performance advantages.
        for name in list(example.keys()):
            tensor = example[name]
            if tensor.dtype == tf.int64:
                tensor = tf.cast(tensor, tf.int32)
            example[name] = tensor
        return example

    fname = config.tfrecord_name.format(
        model_name=config.model_name.replace('/', '_'),
        scope='{scope}',
        index=config.current_fold + 1,
        fold=config.num_k_fold
    )
    logging.info('Parsing TFRecords from {}'.format(fname))
    train_ds = tf.data.TFRecordDataset(fname.format(scope='train')).map(_parse_feature_desc)
    valid_ds = tf.data.TFRecordDataset(fname.format(scope='valid')).map(_parse_feature_desc)
    return train_ds, valid_ds


def configure_tensorflow_dataset(config: Config,
                                 strategy: tf.distribute.MirroredStrategy,
                                 add_steps_per_epoch=False,
                                 add_class_weight=False):
    def _reshape_and_splits_example(example):
        sections = tf.reshape(example['sections'], (-1,))
        labels = tf.reshape(example['labels'], (-1,))
        return example['input_ids'], (sections, labels)

    def _class_weight_map_fn(cw_tuple):
        def _calc_cw(cw):
            class_ids = list(sorted(cw.keys()))
            expected_class_ids = list(range(len(class_ids)))
            if class_ids != expected_class_ids:
                raise ValueError(
                    'Expected `class_weight` to be a dict with keys from 0 to one less'
                    'than the number of classes, found {}'.format(cw)
                )
            return tf.convert_to_tensor([cw[int(c)] for c in class_ids])

        cw_tensors = [_calc_cw(cw_proto) for cw_proto in cw_tuple]

        @tf.function
        def _rearrange_cw(y_true, cw_tensor):
            y_classes = smart_cond.smart_cond(
                y_true.shape.rank == 2 and tf.shape(y_true)[1] > 1,
                lambda: tf.argmax(y_true, axis=1),
                lambda: tf.cast(tf.reshape(y_true, (-1,)), dtype=tf.int32)
            )
            return tf.gather(cw_tensor, y_classes)

        @tf.function
        def _map_fn(input_ids, y_tuple):
            cw = tuple([_rearrange_cw(y_true, cw_tensor) for y_true, cw_tensor in zip(y_tuple, cw_tensors)])
            return input_ids, y_tuple, cw

        return _map_fn

    train_dataset, valid_dataset = parse_tfrecords(config)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE).batch(config.batch_size, drop_remainder=True)
    train_dataset = train_dataset.map(_reshape_and_splits_example)

    steps_per_epoch = 0
    train_labels = None
    for x, ys in train_dataset:
        if train_labels is None:
            train_labels = [[] for _ in range(len(ys))]

        for i, y in enumerate(ys):
            train_labels[i].append(y)

        steps_per_epoch += 1

    class_weights = ()
    for y_batch in train_labels:
        y = np.concatenate(y_batch, axis=0)
        class_weight = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight = dict(enumerate(class_weight))
        class_weights += (class_weight,)

    if add_class_weight:
        train_dataset = train_dataset.map(_class_weight_map_fn(class_weights))

    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat()
    valid_dataset = valid_dataset.batch(config.batch_size).map(_reshape_and_splits_example)

    if config.distribute:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_dataset = strategy.experimental_distribute_dataset(train_dataset.with_options(options))
        valid_dataset = strategy.experimental_distribute_dataset(valid_dataset.with_options(options))

    ds = (train_dataset, valid_dataset)
    if add_steps_per_epoch:
        ds += (steps_per_epoch,)
    if add_class_weight:
        ds += (class_weights,)
    return ds
