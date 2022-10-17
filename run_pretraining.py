import os
import wandb
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import utils, formatting_utils, dataset_utils
from cort.modeling import CortForPretraining
from cort.optimization import GradientAccumulator, create_optimizer
from tensorflow.keras import metrics

formatting_utils.setup_formatter()


@tf.function
def eval_one_step(model, inputs):
    return model(inputs, training=False)


def analyze_representation(model, valid_dataset, val_metric, step):
    num_eval_steps = sum([1 for _ in valid_dataset])
    representations = []
    labels = []

    # Evaluate the model on validation dataset
    for inputs in valid_dataset.take(num_eval_steps):
        loss, eval_outputs = eval_one_step(model, inputs)

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

    df = pd.DataFrame(embeddings, columns=columns)
    df['labels'] = df['labels'].astype(int).astype(str)

    val_loss = val_metric.result().numpy()
    wandb.log({
        'val_loss': val_loss,
        'representations': df
    }, step=step)
    val_metric.reset_state()
    return val_loss


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
    utils.restrict_gpus(config)

    # Initialize W&B agent
    run_name = 'PT-{}_L-{}_I-{}'.format(config.model_name, config.loss_base, utils.generate_random_id())
    wandb.init(project='CoRT Pre-training', name=run_name)

    # Restricting random seed after setting W&B agents
    utils.set_random_seed(config.seed)

    strategy = tf.distribute.MirroredStrategy()
    if config.distribute:
        logging.info('Distributed Training Enabled')

    train_dataset, valid_dataset = dataset_utils.configure_tensorflow_dataset(config, strategy)
    train_iterator = iter(train_dataset)
    with strategy.scope() if config.distribute else utils.empty_context_manager():
        model = CortForPretraining(config)
        accumulator = GradientAccumulator()
        optimizer, learning_rate_fn = create_optimizer(config, config.num_train_steps)
        metric = metrics.Mean(name='loss')
        val_metric = metrics.Mean(name='val_loss')

        checkpoint_dir = os.path.join('./pretraining-checkpoints', wandb.run.id)
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
        num_steps = 0
        while int(checkpoint.step) <= config.num_train_steps:
            step = int(checkpoint.step)
            inputs = next(train_iterator)
            take_step = (num_steps == 0) or (step + 1) % config.gradient_accumulation_steps == 0

            loss = train_one_step(config, model, optimizer, inputs, accumulator, take_step)
            metric.update_state(values=loss)

            # Reports metrics on W&B
            wandb.log({
                'loss': tf.reduce_mean(loss).numpy(),
                'learning_rate': learning_rate_fn(step)
            }, step=step)

            if (step % config.log_freq == 0) and (num_steps % config.gradient_accumulation_steps == 0):
                minutes, seconds = utils.format_minutes_and_seconds(utils.current_milliseconds() - start_time)
                logging.info(
                    'Step: {step:6d}, Loss: {loss:10.6f}, Elapsed: {elapsed}'
                    .format(step=step, loss=metric.result().numpy(),
                            elapsed='{:02d}:{:02d}'.format(minutes, seconds))
                )
                metric.reset_state()
                eval_start_time = utils.current_milliseconds()
                eval_loss = analyze_representation(model, valid_dataset, val_metric, step)
                minutes, seconds = utils.format_minutes_and_seconds(utils.current_milliseconds() - eval_start_time)
                logging.info(
                    ' * Evaluation Loss: {loss:10.6}, Time taken: {taken}'
                    .format(loss=eval_loss,
                            taken='{:02d}:{:02d}'.format(minutes, seconds))
                )

            # Print allreduced metrics on the last step
            if int(checkpoint.step) == config.num_train_steps and num_steps % config.gradient_accumulation_steps == 0:
                minutes, seconds = utils.format_minutes_and_seconds(utils.current_milliseconds() - start_time)
                logging.info(
                    '<FINAL STEP METRICS> Step: {step:6d}, Loss: {loss:10.6f}, Elapsed: {elapsed}'
                    .format(step=step, loss=metric.result().numpy(),
                            elapsed='{:02d}:{:02d}'.format(minutes, seconds))
                )

            if num_steps % config.gradient_accumulation_steps == 0:
                checkpoint.step.assign(int(optimizer.iterations))

            if num_steps % (config.save_checkpoint_steps * config.gradient_accumulation_steps) == 0:
                manager.save(checkpoint_number=step)
                logging.info(' * Saved model checkpoint for step: {}'.format(step))

            num_steps += 1

    logging.info('Finishing all jobs')


if __name__ == '__main__':
    main()
