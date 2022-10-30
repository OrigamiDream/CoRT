import os
import re
import json
import logging
import argparse
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

from cort.config import Config
from cort.modeling import CortForSequenceClassification
from utils import utils, formatting_utils
from tensorflow.keras import metrics
from tensorflow.keras.utils import Progbar
from tensorflow_addons import metrics as metrics_tfa

formatting_utils.setup_formatter(logging.INFO)


KOREAN_PATTERN = re.compile('[ㄱ-ㅎ가-힣]')
CORRELATION_SCORE_UNICODES = ' ▁▂▃▄▅▆▇█'


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

    return dataset, num_steps


@tf.function
def eval_one_step(model, inputs):
    return model(inputs, training=False)


def calc_correlation_from_attentions(input_ids, attentions, tokenizer):
    length = np.sum(input_ids != tokenizer.pad_token_id)

    attention_maps = []
    for attention in attentions:
        reduced = tf.reduce_mean(attention, axis=1)
        reduced = reduced[:, :length - 1, :length - 1]
        attention_maps.append(reduced)

    reduced_attention = tf.concat(attention_maps, axis=0)
    reduced_attention = tf.reduce_mean(reduced_attention, axis=0)

    correlation = reduced_attention[0, 1:]
    correlation = (correlation - tf.reduce_min(correlation)) / (tf.reduce_max(correlation) - tf.reduce_min(correlation))
    return correlation.numpy()


def compose_tokens_correlations(sentence, input_ids, attentions, tokenizer):
    def build_score_unicodes(word_slice, score_index):
        unicodes = []
        for char in word_slice:
            unicode = CORRELATION_SCORE_UNICODES[score_index]
            if KOREAN_PATTERN.match(char):
                unicodes.append(unicode * 2)
            else:
                unicodes.append(unicode)
        return ''.join(unicodes)

    def colorize(text, attention_score, c1=(255, 0, 0), c2=(0, 255, 0)):
        color = (1 - attention_score) * np.array(list(c1)) + attention_score * np.array(list(c2))
        return '\033[38;2;{};{};{}m{}\033[0m'.format(int(color[0]), int(color[1]), int(color[2]), text)

    ComposedToken = collections.namedtuple('ComposedToken', [
        'matched', 'text', 'colorized_text', 'token', 'token_index', 'correlation_score', 'correlation_unicode'
    ])

    length = np.sum(input_ids != tokenizer.pad_token_id)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, 1:length - 1])  # remove [CLS], [SEP], [PAD]
    correlations = calc_correlation_from_attentions(input_ids, attentions, tokenizer)

    offset = 0
    maxlen = len(sentence)
    composed_tokens = []
    for i, token in enumerate(tokens):
        while True:
            matched = True
            if token.startswith('##'):
                matched = sentence[offset - 1] != ' '  # previous letter must not be space
                token = token[2:]

            word = sentence[offset:offset + min(len(token), maxlen)]
            matched = matched and token == word.lower()

            if matched:
                offset += len(token)
                score = correlations[i]
                unicode_index = int(score * (len(CORRELATION_SCORE_UNICODES) - 1))

                composed_tokens.append(ComposedToken(
                    matched=True,
                    text=word,
                    colorized_text=colorize(word, score),
                    token=token,
                    token_index=i,
                    correlation_score=score,
                    correlation_unicode=build_score_unicodes(word, unicode_index)
                ))
                break
            else:
                offset += 1
                composed_tokens.append(ComposedToken(
                    matched=False,
                    text=' ',
                    colorized_text=' ',
                    token=None,
                    token_index=-1,
                    correlation_score=-1,
                    correlation_unicode=' '
                ))
    return composed_tokens


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
        name='macro_f1_score', num_classes=config.num_labels, average='macro')

    if config.repr_finetune:
        metric_map['co_loss'] = metrics.Mean(name='co_loss')
        metric_map['cce_loss'] = metrics.Mean(name='cce_loss')
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


def perform_interactive_predictions(config, model):
    try:
        from cort.preprocessing import run_multiprocessing_job, normalize_texts, LABEL_NAMES
    except ImportError as e:
        logging.error(e)
        return

    tokenizer = utils.create_tokenizer_from_config(config)
    if hasattr(tokenizer, 'disable_progressbar'):
        tokenizer.disable_progressbar = True

    print('You can perform inference interactively here, `q` to end the process')
    while True:
        sentence = input('\nSentence: ')
        if sentence == 'q':
            break

        orig = sentence
        sentence = normalize_texts(sentence)
        sentence = sentence.lower()

        tokenized = tokenizer([sentence],
                              padding='max_length',
                              truncation=True,
                              return_attention_mask=False,
                              return_token_type_ids=False)
        input_ids = np.array(tokenized['input_ids'], dtype=np.int32)
        _, cort_outputs = model(input_ids, training=False)

        composed_tokens = compose_tokens_correlations(orig, input_ids, cort_outputs['attentions'], tokenizer)

        probs = cort_outputs['probs'][0]
        index = np.argmax(probs)
        print('\nCorrelations:')
        print(''.join([composed.correlation_unicode for composed in composed_tokens]))
        print(''.join([composed.colorized_text for composed in composed_tokens]))
        print('\nPrediction: {}: ({:.06f} of confidence score)'.format(LABEL_NAMES[index], probs[index]))
        print()


def perform_inference(args, config, model):
    dataset, num_steps = parse_tfrecords(args)

    metric_maps = create_metric_map(config)
    compile_metric_names = ['accuracy', 'recall', 'precision', 'micro_f1_score', 'macro_f1_score']
    model.compile(loss=model.loss_fn, metrics=[metric_maps[name] for name in compile_metric_names])

    logging.info('***** Inference *****')
    logging.info('  Model name = {}'.format(args.model_name))
    logging.info('  Batch size = {}'.format(args.batch_size))
    logging.info('  Repr classifier = {}'.format(args.repr_classifier))
    logging.info('  Repr activation = {}'.format(args.repr_act))
    logging.info('  Num of concatenating hidden states = {}'.format(args.concat_hidden_states))

    progbar = Progbar(num_steps, stateful_metrics=[metric.name for metric in metric_maps.values()])
    representations = []
    labels = []
    for step, inputs in enumerate(dataset):
        loss, cort_outputs = eval_one_step(model, inputs)

        metric_maps['loss'].update_state(values=loss)
        metric_fn(metric_maps, cort_outputs, config)

        if 'representation' in cort_outputs:
            representations.append(cort_outputs['representation'].numpy())
            labels.append(cort_outputs['labels'].numpy())

        progbar.update(step, values=[
            (metric_name, float(metric.result().numpy())) for metric_name, metric in metric_maps.items()
        ])
    progbar.update(
        current=num_steps,
        values=[(metric_name, float(metric.result().numpy())) for metric_name, metric in metric_maps.items()],
        finalize=True
    )
    logging.info('***** Evaluation results *****')
    for metric_name, metric in metric_maps.items():
        logging.info('  {}: {:.06f}'.format(metric_name, metric.result().numpy()))

    os.makedirs(args.outputs_dir, exist_ok=True)

    model_name = args.model_name.replace('/', '_')
    repr_table = create_scatter_representation_table(representations, labels)
    fname = os.path.join(args.outputs_dir, '{}_representations.csv'.format(model_name))
    repr_table.to_csv(fname, index=False)
    logging.info('Exported representations to {}'.format(fname))

    fname = os.path.join(args.outputs_dir, '{}_eval_outputs.json'.format(model_name))
    body = {
        metric_name: float(metric.result().numpy()) for metric_name, metric in metric_maps.items()
    }
    with open(fname, 'w') as f:
        json.dump(body, f, indent=4, sort_keys=True)
    logging.info('Exported eval outputs to {}'.format(fname))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--checkpoint_path', required=True,
                        help='Location of trained model checkpoint.')
    parser.add_argument('--model_name', default='klue/roberta-base',
                        help='Name of pre-trained models. (One of korscibert, korscielectra, huggingface models)')
    parser.add_argument('--tfrecord_path', default='./data/tfrecords/{model_name}/eval.tfrecord',
                        help='Location of TFRecord file for inference. {model_name} is a placeholder.')
    parser.add_argument('--outputs_dir', default='./eval-outputs',
                        help='Location of results from model inference')
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
    parser.add_argument('--interactive', default=False, type=bool,
                        help='Interactive mode for real-time inference')

    # Configurable pre-defined variables
    parser.add_argument('--korscibert_vocab', default='./cort/pretrained/korscibert/vocab_kisti.txt')
    parser.add_argument('--korscibert_ckpt', default='./cort/pretrained/korscibert/model.ckpt-262500')
    parser.add_argument('--korscielectra_vocab', default='./cort/pretrained/korscielectra/data/vocab.txt')
    parser.add_argument('--korscielectra_ckpt', default='./cort/pretrained/korscielectra/data/models/korsci_base')
    parser.add_argument('--repr_finetune', default=True, type=bool)
    parser.add_argument('--repr_preact', default=True, type=bool)
    parser.add_argument('--loss_base', default='supervised')
    parser.add_argument('--classifier_dropout_prob', default=0.1, type=float)
    parser.add_argument('--backbone_trainable_layers', default=0, type=float)

    # Parse arguments
    args = parser.parse_args()
    config = Config(**vars(args))
    config.pretrained_config = utils.parse_pretrained_config(config)

    model = CortForSequenceClassification(config, num_labels=config.num_labels)
    model.trainable = False

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.checkpoint_path).expect_partial()
    logging.info('Restored model checkpoint from {}'.format(args.checkpoint_path))

    if args.interactive:
        logging.info('Interactive mode enabled')
        perform_interactive_predictions(config, model)
    else:
        perform_inference(args, config, model)
    logging.info('Finishing all jobs')


if __name__ == '__main__':
    main()
