import os
import logging
import argparse

import tensorflow as tf

from utils import utils, formatting_utils
from cort.config import Config
from cort.modeling import CortForSequenceClassification
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.prediction_log_pb2 import PredictionLog, PredictLog

formatting_utils.setup_formatter(logging.INFO)


def parse_tfrecords(tfrecord_path, model_name, maxlen, num_samples):
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

    def _reconfigure_inputs(example):
        return example['input_ids']

    fname = tfrecord_path.format(model_name=model_name.replace('/', '_'))
    logging.info('Parsing TFRecords from {}'.format(fname))

    dataset = tf.data.TFRecordDataset(fname)
    dataset = dataset.map(_parse_feature_desc).map(_reconfigure_inputs)
    dataset = dataset.shuffle(buffer_size=1024).repeat().batch(num_samples)

    input_ids = None
    for input_ids in dataset:
        break
    return input_ids


def store_warmup_requests(args, input_ids, saved_model_path):
    warmup_request_dir = os.path.join(saved_model_path, 'assets.extra')
    os.makedirs(warmup_request_dir, exist_ok=True)
    warmup_request_path = os.path.join(warmup_request_dir, 'tf_serving_warmup_requests')

    with tf.io.TFRecordWriter(warmup_request_path) as writer:
        input_ids = tf.make_tensor_proto(input_ids)

        request = PredictRequest()
        request.model_spec.name = args.model_spec_name
        request.model_spec.signature_name = args.signature_name
        request.inputs['input_ids'].CopyFrom(input_ids)

        log = PredictionLog(predict_log=PredictLog(request=request))
        writer.write(log.SerializeToString())
    logging.info('{} warmup requests have been stored at: {}'.format(args.num_warmup_requests, warmup_request_path))


def restore_cort_classifier(args, config: Config):
    cort_model = CortForSequenceClassification(config, num_labels=config.num_labels)
    cort_model.trainable = False

    # Restore from checkpoint
    checkpoint = tf.train.Checkpoint(model=cort_model)
    checkpoint.restore(args.checkpoint_path).expect_partial()

    serving = CortForSequenceClassification.Serving(config, cort_model)
    serving(serving.dummy_inputs)
    logging.info('Restored model checkpoint from: {}'.format(args.checkpoint_path))
    return serving


def store_as_saved_model(cort_model, signature_name, filepath):
    maxlen = cort_model.config.pretrained_config.max_position_embeddings

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, maxlen), dtype=tf.int32, name='input_ids')])
    def _eval_wrapper(input_ids):
        return cort_model(input_ids)

    signatures = _eval_wrapper.get_concrete_function()
    tf.saved_model.save(cort_model, filepath, signatures={
        signature_name: signatures
    })
    logging.info('Servable CoRT classifier has been written as SavedModel format at: {}'.format(filepath))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--checkpoint_path', required=True,
                        help='Location of trained model checkpoint.')
    parser.add_argument('--saved_model_dir', default='./models',
                        help='Location of SavedModel to be stored.')
    parser.add_argument('--model_spec_name', default='cort',
                        help='Name of model spec.')
    parser.add_argument('--model_spec_version', default='1',
                        help='Version of model spec.')
    parser.add_argument('--signature_name', default='serving_default',
                        help='Name of signature of SavedModel')
    parser.add_argument('--model_name', default='klue/roberta-base',
                        help='Name of pre-trained models. (One of korscibert, korscielectra, huggingface models)')
    parser.add_argument('--tfrecord_path', default='./data/tfrecords/{model_name}/eval.tfrecord',
                        help='Location of TFRecord file for warmup requests. {model_name} is a placeholder.')
    parser.add_argument('--num_warmup_requests', default=10, type=int,
                        help='Number of warmup requests. Pass 0 to skip')
    parser.add_argument('--repr_classifier', default='seq_cls',
                        help='Name of classification head for classifier. (One of seq_cls and bi_lstm is allowed)')
    parser.add_argument('--repr_act', default='tanh',
                        help='Name of activation function for representation. (One of tanh and gelu is allowed)')
    parser.add_argument('--concat_hidden_states', default=1, type=int,
                        help='Number of hidden states to concatenate.')
    parser.add_argument('--repr_size', default=1024, type=int,
                        help='Number of representation dense units')
    parser.add_argument('--num_labels', default=9, type=int,
                        help='Number of labels')

    # Configurable pre-defined variables
    parser.add_argument('--korscibert_vocab', default='./cort/pretrained/korscibert/vocab_kisti.txt')
    parser.add_argument('--korscibert_ckpt', default='./cort/pretrained/korscibert/model.ckpt-262500')
    parser.add_argument('--korscielectra_vocab', default='./cort/pretrained/korscielectra/data/vocab.txt')
    parser.add_argument('--korscielectra_ckpt', default='./cort/pretrained/korscielectra/data/models/korsci_base')
    parser.add_argument('--classifier_dropout_prob', default=0.1, type=float)

    # Parser arguments
    args = parser.parse_args()
    config = Config(**vars(args))
    config.pretrained_config = utils.parse_pretrained_config(config)
    saved_model_path = os.path.join(args.saved_model_dir, args.model_spec_name, args.model_spec_version)

    cort_serving = restore_cort_classifier(args, config)

    store_as_saved_model(cort_serving, args.signature_name, saved_model_path)

    if args.num_warmup_requests > 0:
        maxlen = config.pretrained_config.max_position_embeddings
        input_ids = parse_tfrecords(args.tfrecord_path, args.model_name, maxlen, num_samples=args.num_warmup_requests)
        store_warmup_requests(args, input_ids, saved_model_path)

    logging.info('Finishing all necessary jobs')
    logging.info('Run following command to build and run Docker container:')
    logging.info(
        '  MODEL_DIR={} MODEL_NAME={} MODEL_VERSION={} docker build -t cort/serving:latest .'
        .format(args.saved_model_dir,
                args.model_spec_name,
                args.model_spec_version)
    )
    logging.info('  docker run -d -p 8500:8500 --name cort-serving cort/serving')


if __name__ == '__main__':
    main()
