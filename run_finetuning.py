import os
import wandb
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from cort.modeling import CortForSequenceClassification, CortForElaboratedSequenceClassification, CortForPretraining
from cort.optimization import create_optimizer
from utils import utils, formatting_utils, dataset_utils
from tensorflow.keras import metrics
from tensorflow.keras.utils import Progbar
from tensorflow_addons import metrics as metrics_tfa

formatting_utils.setup_formatter()


@tf.function
def train_one_step(model, optimizer, inputs, clip_norm=1.0):
    with tf.GradientTape() as tape:
        loss, cort_outputs = model(inputs, training=True)
    grads = tape.gradient(loss, model.trainable_variables)

    (gradients, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, cort_outputs


@tf.function
def eval_one_step(model, inputs):
    return model(inputs, training=False)


def create_scatter_representation_table(representations, labels):
    representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = np.reshape(labels, (-1, 1))

    embedding_size = representations.shape[1]
    columns = ['labels'] + ['e{}'.format(i) for i in range(embedding_size)]
    embeddings = np.concatenate([labels, representations], axis=-1)

    df = pd.DataFrame(embeddings, columns=columns)
    df['labels'] = df['labels'].astype(int).astype(str)

    return df


def create_pretrained_replica(config, ckpt_path):
    replica = CortForPretraining(config)
    replica(replica.dummy_inputs)

    optimizer, _ = create_optimizer(config, 10000)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=replica)
    checkpoint.restore(ckpt_path)  # unresolved optimizer variables warnings
    return replica


def create_metric_map(config):
    metric_map = dict()
    metric_map['loss'] = metrics.Mean(name='loss')
    metric_map['accuracy'] = metrics.CategoricalAccuracy(name='accuracy')
    metric_map['recall'] = metrics.Recall(name='recall')
    metric_map['precision'] = metrics.Precision(name='precision')
    metric_map['micro_f1_score'] = metrics_tfa.F1Score(
        name='micro_f1_score', num_classes=config.num_labels, average='micro'
    )
    metric_map['macro_f1_score'] = metrics_tfa.F1Score(
        name='macro_f1_score', num_classes=config.num_labels, average='macro'
    )

    # Metrics for model without Pre-trained model
    if config.repr_finetune:
        metric_map['co_loss'] = metrics.Mean(name='co_loss')
        metric_map['cce_loss'] = metrics.Mean(name='cce_loss')

    # Metrics for model training from elaborated representation
    if config.include_sections:
        metric_map['section_accuracy'] = metrics.CategoricalAccuracy(name='section_accuracy')
        metric_map['section_recall'] = metrics.Recall(name='section_recall')
        metric_map['section_precision'] = metrics.Precision(name='section_precision')
        metric_map['section_micro_f1_score'] = metrics_tfa.F1Score(
            name='section_micro_f1_score', num_classes=config.num_sections, average='micro'
        )
        metric_map['section_macro_f1_score'] = metrics_tfa.F1Score(
            name='section_macro_f1_score', num_classes=config.num_sections, average='macro'
        )
        metric_map['section_co_loss'] = metrics.Mean(name='section_co_loss')
        metric_map['section_cce_loss'] = metrics.Mean(name='section_cce_loss')

    return metric_map


def metric_fn(dicts, cort_outputs, config):
    d = cort_outputs
    confusion_keys = ['accuracy', 'recall', 'precision',
                      'micro_f1_score', 'macro_f1_score']
    for key in confusion_keys:
        dicts[key].update_state(
            y_true=d['ohe_labels'],
            y_pred=d['probs']
        )

    if config.repr_finetune:
        dicts['co_loss'].update_state(values=d['co_loss'])
        dicts['cce_loss'].update_state(values=d['cce_loss'])

    if config.include_sections:
        dicts['section_co_loss'].update_state(values=d['section_co_loss'])
        dicts['section_cce_loss'].update_state(values=d['section_cce_loss'])
        confusion_keys = ['section_' + key for key in confusion_keys]
        for key in confusion_keys:
            dicts[key].update_state(
                y_true=d['section_ohe_labels'],
                y_pred=d['section_probs']
            )


def main():
    config = utils.parse_arguments()
    utils.restrict_gpus(config)

    # Initialize W&B agent
    if not config.train_at_once and not config.pretraining_run_name:
        raise ValueError('Pre-training run name must be provided when uses Pre-trained models')
    elif not config.train_at_once and not config.pretraining_checkpoint_dir:
        raise ValueError('Pre-training checkpoint dir path must be provided when uses Pre-trained models')
    elif not config.train_at_once and config.pretraining_run_name and config.pretraining_checkpoint_dir:
        pretrained_run_name = config.pretraining_run_name
        if config.checkpoint_dir:
            raise ValueError('Directory to checkpoint must be empty when uses Pre-trained models')
        config.checkpoint_dir = config.pretraining_checkpoint_dir.format(run_name=pretrained_run_name)
        run_name = 'FT-{}_P-{}_I-{}'.format(config.model_name, pretrained_run_name, utils.generate_random_id())
    else:
        run_name = 'FT-{}_P-None_I-{}'.format(config.model_name, utils.generate_random_id())

    wandb.init(project='CoRT Fine-tuning', name=run_name)

    # Restricting random seed after setting W&B agents
    utils.set_random_seed(config.seed)

    strategy = tf.distribute.MirroredStrategy()
    if config.distribute:
        logging.info('Distributed Training Enabled')

    train_dataset, valid_dataset, steps_per_epoch, _ = dataset_utils.configure_tensorflow_dataset(
        config, strategy, add_steps_per_epoch=True, add_class_weight=True
    )
    total_train_steps = config.epochs * steps_per_epoch
    logging.info('Training steps_per_epoch: {}, total_train_steps: {}'.format(steps_per_epoch, total_train_steps))
    with strategy.scope() if config.distribute else utils.empty_context_manager():
        if config.repr_finetune and config.include_sections:
            logging.info('Fine-tuning Representation, Including Sections → Elaborated Representation')
            model = CortForElaboratedSequenceClassification(
                config,
                num_sections=config.num_sections,
                num_labels=config.num_labels
            )
        else:
            logging.info('Excluding Sections → Label Representation')
            model = CortForSequenceClassification(config, num_labels=config.num_labels)

            # Freeze the Pre-trained CoRT encoder for 2-stage training
            if not config.train_at_once:
                model.cort.trainable = False
                logging.info('Froze CoRT encoder layers')

        checkpoint_dir = os.path.join('./finetuning-checkpoints', wandb.run.id)
        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=config.keep_checkpoint_max)
        if config.restore_checkpoint and config.restore_checkpoint == 'latest' and not config.pretraining_run_name:
            checkpoint.restore(manager.latest_checkpoint)
            logging.info('Restored latest model checkpoint from {}'.format(config.checkpoint_dir))
        elif config.restore_checkpoint and config.restore_checkpoint != 'latest' and not config.pretraining_run_name:
            checkpoint.restore(config.restore_checkpoint)
            logging.info('Restored specified model checkpoint from {}'.format(config.checkpoint_dir))
        elif config.restore_checkpoint and config.pretraining_run_name:
            title = '#### Restoring Pre-trained Models ####'
            logging.info('#' * len(title))
            logging.info(title)
            if config.restore_checkpoint == 'latest':
                checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)
            else:
                checkpoint_path = os.path.join(config.checkpoint_dir, config.restore_checkpoint)
            pretrained_replica = create_pretrained_replica(config, checkpoint_path)
            model.cort.set_weights(pretrained_replica.cort.get_weights())
            logging.info('Restored Pre-trained `{}` model from {}'.format(config.model_name, config.checkpoint_dir))
            logging.info('#' * len(title))
        else:
            logging.info('Initializing from scratch')

        metric_maps = create_metric_map(config)
        compile_metric_names = ['accuracy', 'recall', 'precision', 'micro_f1_score', 'macro_f1_score']
        optimizer, learning_rate_fn = create_optimizer(config, total_train_steps)

        model.compile(
            optimizer=optimizer, loss=model.loss_fn,
            metrics=[metric_maps[name] for name in compile_metric_names]
        )

        logging.info('***** Running training *****')
        logging.info('  Num examples = {}'.format(steps_per_epoch * config.batch_size))
        logging.info('  Num epochs = {}'.format(config.epochs))
        logging.info('  Batch size = {}'.format(config.batch_size))
        logging.info('  Total training steps = {}'.format(total_train_steps))

        num_steps = 0
        for epoch in range(config.initial_epoch, config.epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, config.epochs))
            progbar = Progbar(steps_per_epoch, stateful_metrics=[metric.name for metric in metric_maps.values()])

            # Forward and Backprop
            for step, inputs in enumerate(train_dataset.take(steps_per_epoch)):
                loss, cort_outputs = train_one_step(model, optimizer, inputs)

                # Update metrics with model outputs
                metric_maps['loss'].update_state(values=loss)
                metric_fn(metric_maps, cort_outputs, config)
                progbar.update(step, values=[
                    (metric_name, float(metric.result().numpy())) for metric_name, metric in metric_maps.items()
                ])
                # Reports metrics on W&B
                wandb.log({
                    'loss': tf.reduce_mean(loss).numpy(),
                    'iterations': optimizer.iterations.numpy(),
                    'learning_rate': learning_rate_fn(optimizer.iterations)
                }, step=num_steps)
                wandb.log({
                    metric_name: metric.result().numpy() for metric_name, metric in metric_maps.items()
                }, step=num_steps)
                num_steps += 1

            # Reset all metric states for evaluation
            epoch_logs = {}
            for metric_name, metric in metric_maps.items():
                epoch_logs[metric_name] = float(metric.result().numpy())
                metric.reset_state()

            # Evaluation
            representations = []
            labels = []
            section_representations = []
            sections = []
            for step, inputs in enumerate(valid_dataset):
                loss, cort_outputs = eval_one_step(model, inputs)

                # Update metrics with model outputs
                metric_maps['loss'].update_state(values=loss)
                metric_fn(metric_maps, cort_outputs, config)

                if 'representation' in cort_outputs:
                    representations.append(cort_outputs['representation'].numpy())
                    labels.append(cort_outputs['labels'].numpy())

                if 'section_representation' in cort_outputs:
                    section_representations.append(cort_outputs['section_representation'].numpy())
                    sections.append(cort_outputs['sections'].numpy())

            wandb_logs = {}
            for metric_name, metric in metric_maps.items():
                value = float(metric.result().numpy())
                epoch_logs['val_' + metric_name] = value
                wandb_logs['val_' + metric_name] = value
                metric.reset_state()

            if len(representations) > 0:
                wandb_logs['representations'] = create_scatter_representation_table(representations, labels)
            if len(section_representations) > 0:
                wandb_logs['section_representations'] = create_scatter_representation_table(
                    section_representations, sections
                )
            # Reports evaluation results on W&B
            wandb.log(wandb_logs, step=num_steps)

            progbar.update(
                current=steps_per_epoch,
                values=[(name, value) for name, value in epoch_logs.items()],
                finalize=True
            )
            manager.save(checkpoint_number=epoch)
            logging.info(' * Saved model checkpoint for epoch: {}'.format(epoch))

    logging.info('Finishing all jobs')


if __name__ == '__main__':
    main()
