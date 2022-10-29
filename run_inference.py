import logging
import argparse

import tensorflow as tf

from cort.config import Config
from cort.modeling import CortForSequenceClassification
from utils import utils, formatting_utils
from tensorflow.keras import metrics
from tensorflow_addons import metrics as metrics_tfa

formatting_utils.setup_formatter(logging.INFO)


def parse_tfrecords(args):
    feature_desc = {
        'input_ids': tf.io.FixedLenFeature([args.max_position_embeddings], tf.int64),
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

    def _reshape_and_splits_example(example):
        sections = tf.reshape(example['sections'], (-1,))
        labels = tf.reshape(example['labels'], (-1,))
        return example['input_ids'], (sections, labels)

    fname = args.tfrecord_path.format(model_name=args.model_name.replace('/', '_'))
    logging.info('Parsing TFRecords from {}'.format(fname))

    dataset = tf.data.TFRecordDataset(fname).map(_parse_feature_desc).batch(args.batch_size)
    dataset = dataset.map(_reshape_and_splits_example)

    num_steps = 0
    for _ in dataset:
        num_steps += 1

    num_steps *= args.batch_size
    return dataset, num_steps


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--checkpoint_path', required=True,
                        help='Location of trained model checkpoint.')
    parser.add_argument('--model_name', default='klue/roberta-base',
                        help='Name of pre-trained models. (One of korscibert, korscielectra, huggingface models)')
    parser.add_argument('--tfrecord_path', default='./data/tfrecords/{model_name}/eval.tfrecord',
                        help='Location of TFRecord file for inference. {model_name} is a placeholder.')
    parser.add_argument('--repr_classifier', default='seq_cls',
                        help='Name of classifier head for classifier. (One of seq_cls and bi_lstm is allowed)')
    parser.add_argument('--repr_act', default='tanh',
                        help='Name of activation function for representation. (One of tanh and gelu is allowed)')
    parser.add_argument('--concat_hidden_states', default=1, type=int,
                        help='Number of hidden states to concatenate.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Number of batch size.')
    parser.add_argument('--max_position_embeddings', default=512, type=int,
                        help='Number of maximum position embeddings.')
    parser.add_argument('--repr_size', default=1024, type=int,
                        help='Number of representation dense units')
    parser.add_argument('--num_labels', default=9, type=int,
                        help='Number of labels')
    parser.add_argument('--seed', default=42, type=int,
                        help='A seed of random state.')

    # Configurable pre-defined variables
    parser.add_argument('--korscibert_vocab', default='./cort/pretrained/korscibert/vocab_kisti.txt')
    parser.add_argument('--korscibert_ckpt', default='./cort/pretrained/korscibert/model.ckpt-262500')
    parser.add_argument('--korscielectra_vocab', default='./cort/pretrained/korscielectra/data/vocab.txt')
    parser.add_argument('--korscielectra_ckpt', default='./cort/pretrained/korscielectra/data/models/korsci_base')
    parser.add_argument('--repr_preact', default=True, type=bool)
    parser.add_argument('--classifier_dropout_prob', default=0.1, type=float)
    parser.add_argument('--backbone_trainable_layers', default=0, type=float)

    # Parse arguments
    args = parser.parse_args()
    config = Config(**vars(args))
    config.pretrained_config = utils.parse_pretrained_config(config)

    dataset, num_steps = parse_tfrecords(args)

    model = CortForSequenceClassification(config, num_labels=config.num_labels)
    model.trainable = False

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.checkpoint_path)
    logging.info('Restored model checkpoint from {}'.format(args.checkpoint_path))

    metrics_array = [
        metrics.CategoricalAccuracy(name='accuracy'),
        metrics.Recall(name='recall'),
        metrics.Precision(name='precision'),
        metrics_tfa.F1Score(name='micro_f1_score', num_classes=config.num_labels, average='micro'),
        metrics_tfa.F1Score(name='macro_f1_score', num_classes=config.num_labels, average='macro')
    ]
    model.compile(loss=model.loss_fn, metrics=metrics_array)

    logging.info('***** Inference *****')
    logging.info('  Model name = {}'.format(args.model_name))
    logging.info('  Batch size = {}'.format(args.batch_size))
    logging.info('  Repr classifier = {}'.format(args.repr_classifier))
    logging.info('  Repr activation = {}'.format(args.repr_act))
    logging.info('  Num of concatenating hidden states = {}'.format(args.concat_hidden_states))

    outputs = model.evaluate(dataset, steps=num_steps)

    cce_loss = outputs[0]
    metrics_map = {
        metric.name: output for output, metric in zip(outputs[1:], metrics_array)
    }
    logging.info('***** Evaluation results *****')
    logging.info('  loss: {:.06f}'.format(cce_loss))
    for metric_name, value in metrics_map.items():
        logging.info('  {}: {:.06f}'.format(metric_name, value))


if __name__ == '__main__':
    main()
