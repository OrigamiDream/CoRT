import os
import wandb
import logging
import numpy as np
import tensorflow as tf

from utils import utils, formatting_utils
from cort.config import Config
from cort.modeling import CortForPretraining
from cort.optimization import GradientAccumulator, create_optimizer
from tensorflow.keras import metrics

formatting_utils.setup_formatter()


def restrict_gpus(config: Config):
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        logging.warning('No available GPUs')
        return

    if config.gpu != 'all':
        desired_gpu = gpus[int(config.gpu)]
        tf.config.set_visible_devices(desired_gpu, 'GPU')
        logging.info('Restricting GPU as /device:GPU:{}'.format(config.gpu))


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


def analyze_representation(model, valid_dataset, val_metric):
    representations = []
    labels = []

    # Evaluate the model on validation dataset
    for index, inputs in enumerate(valid_dataset):
        loss, eval_outputs = model(inputs, training=False)

        representations.append(eval_outputs['representation'].numpy())
        labels.append(eval_outputs['labels'].numpy())

        val_metric.update_state(values=loss)

    representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = np.reshape(labels, (-1, 1))

    # Reports metrics and representation on W&B
    embedding_size = representations.shape[1]
    columns = ['labels'] + ['embed_{}'.format(i) for i in range(embedding_size)]
    embeddings = np.concatenate([labels, representations], axis=-1)
    wandb.log({
        'val_loss': val_metric.result().numpy(),
        'representations': wandb.Table(
            columns=columns,
            data=embeddings
        )
    })
    val_metric.reset_state()


@tf.function
def train_one_step(config, model, optimizer, inputs, accumulator, take_step, clip_norm=1.0):
    # Forward and backprop
    with tf.GradientTape() as tape:
        loss, _ = model(inputs, training=True)
    grads = tape.gradient(loss, model.trainable_variables)

    # Accumulate gradients
    accumulator(grads)
    if take_step:
        # All reduce and clip the accumulated gradients
        reduced_accumulated_gradients = [
            None if g is None else g / tf.cast(config.gradient_accumulation_steps, g.dtype)
            for g in accumulator.accumulated_gradients
        ]
        (clipped_accumulated_gradients, _) = tf.clip_by_global_norm(reduced_accumulated_gradients, clip_norm=clip_norm)

        # Weight update
        optimizer.apply_gradients(zip(clipped_accumulated_gradients, model.trainable_variables))
        accumulator.reset()

    return loss


def main():
    config = utils.parse_arguments()
    restrict_gpus(config)
    utils.set_random_seed(config.seed)

    # Initialize W&B agent
    run_name = 'fold-{}_{}_{}'.format(config.current_fold + 1, config.model_name, utils.generate_random_id())
    wandb.init(project='CoRT Pre-training', name=run_name)

    strategy = tf.distribute.MirroredStrategy()
    if config.distribute:
        logging.info('Distributed Training Enabled')

    def _reshape_and_splits_example(example):
        sections = tf.reshape(example['sections'], (-1,))
        labels = tf.reshape(example['labels'], (-1,))
        return example['input_ids'], (sections, labels)

    train_dataset, valid_dataset = parse_tfrecords(config)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE).batch(config.batch_size, drop_remainder=True)
    train_dataset = train_dataset.map(_reshape_and_splits_example)
    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat()

    valid_dataset = valid_dataset.batch(config.batch_size).map(_reshape_and_splits_example)

    if config.distribute:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_dataset = strategy.experimental_distribute_dataset(train_dataset.with_options(options))
        valid_dataset = strategy.experimental_distribute_dataset(valid_dataset.with_options(options))

    with strategy.scope() if config.distribute else utils.empty_context_manager():
        model = CortForPretraining(config)
        accumulator = GradientAccumulator()
        optimizer, learning_rate_fn = create_optimizer(config, config.num_train_steps)
        metric = metrics.Mean(name='loss')
        val_metric = metrics.Mean(name='val_loss')

        checkpoint_dir = os.path.join('./checkpoints', wandb.run.id)
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=config.keep_checkpoint_max)
        if config.restore_checkpoint and config.restore_checkpoint != 'latest':
            checkpoint.restore(config.restore_checkpoint)
            logging.info('Restored model checkpoint from {}'.format(config.restore_checkpoint))
        elif config.restore_checkpoint and config.restore_checkpoint == 'latest':
            checkpoint.restore(manager.latest_checkpoint)
            logging.info('Restored model checkpoint from {}'.format(manager.latest_checkpoint))
        else:
            logging.info('Initializing from scratch')

        accumulator.reset()
        start_time = utils.current_milliseconds()
        num_steps = int(checkpoint.step)
        for inputs in train_dataset.skip(num_steps):
            step = int(checkpoint.step)
            take_step = (step == 0) or (step + 1) % config.gradient_accumulation_steps == 0

            loss = train_one_step(config, model, optimizer, inputs, accumulator, take_step)
            metric.update_state(values=loss)

            # Reports metrics on W&B
            wandb.log({
                'loss': metric.result().numpy(),
                'learning_rate': learning_rate_fn(num_steps)
            })

            if (step % config.log_freq == 0) and (num_steps % config.gradient_accumulation_steps == 0):
                minutes, seconds = utils.format_minutes_and_seconds(utils.current_milliseconds() - start_time)
                logging.info(
                    'Step: {step:6d}, Loss: {loss:10.6f}, Elapsed: {elapsed}'
                    .format(step=step, loss=metric.result().numpy(),
                            elapsed='{:02d}:{:02d}'.format(minutes, seconds))
                )
                metric.reset_state()
                analyze_representation(model, valid_dataset, val_metric)

            if num_steps % config.gradient_accumulation_steps == 0:
                checkpoint.step.assign(int(optimizer.iterations))

            if num_steps % (config.save_checkpoint_steps * config.gradient_accumulation_steps) == 0:
                manager.save(checkpoint_number=step)
                logging.info('Saved model checkpoint for step: {}'.format(step))

            num_steps += 1

            if num_steps == config.num_train_steps:
                minutes, seconds = utils.format_minutes_and_seconds(utils.current_milliseconds() - start_time)
                logging.info(
                    '<FINAL STEP METRICS> Step: {step:6d}, Loss: {loss:10.6f}, Elapsed: {elapsed}'
                    .format(step=step, loss=metric.result().numpy(),
                            elapsed='{:02d}:{:02d}'.format(minutes, seconds))
                )
                metric.reset_state()

    # Finishing W&B agent
    wandb.finish()


if __name__ == '__main__':
    main()
