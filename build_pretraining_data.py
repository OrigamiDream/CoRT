import os
import shutil
import logging
import argparse
import numpy as np
import tensorflow as tf

from utils import utils
from utils.formatting_utils import setup_formatter
from cort.preprocessing import parse_and_preprocess_sentences, run_multiprocessing_job, normalize_texts
from sklearn.model_selection import StratifiedKFold, train_test_split


def preprocess_sentences_on_batch(batch):
    sentences = []
    for sentence in batch:
        sentence = normalize_texts(sentence, remove_specials=False, remove_last_period=False)
        sentence = sentence.lower()  # do_lower_case
        sentences.append(sentence)
    return sentences


def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def write_examples(fname, input_ids, sections, labels, indices=None):
    writer = tf.io.TFRecordWriter(fname)

    if indices is not None:
        input_ids = input_ids[indices]
        sections = sections[indices]
        labels = labels[indices]

    assert len(input_ids) == len(sections) == len(labels), (
        'Number of all samples must be same'
    )
    for input_id, section, label in zip(input_ids, sections, labels):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'input_ids': create_int_feature(input_id),
            'sections': create_int_feature([section]),
            'labels': create_int_feature([label])
        }))
        writer.write(tf_example.SerializeToString())
    logging.info('{} examples written at: {}'.format(len(input_ids), fname))
    writer.close()


def create_tfrecords(args, output_dir):
    logging.info('Normalizing examples')
    df = parse_and_preprocess_sentences(args.filepath)
    results = run_multiprocessing_job(preprocess_sentences_on_batch, df['sentences'],
                                      num_processes=args.num_processes)
    sentences = []
    for batch in results:
        sentences += batch

    logging.info('Tokenizing examples')
    tokenizer = utils.create_tokenizer_from_config(args)
    tokenized = tokenizer(sentences,
                          padding='max_length',
                          truncation=True,
                          return_attention_mask=False,
                          return_token_type_ids=False)
    input_ids = np.array(tokenized['input_ids'], dtype=np.int32)
    sections = np.array(df['code_sections'].values, dtype=np.int32)
    labels = np.array(df['code_labels'].values, dtype=np.int32)

    if args.test_size and args.test_size < 1.0:
        splits = train_test_split(input_ids, sections, labels,
                                  test_size=args.test_size, random_state=args.seed, shuffle=True, stratify=labels)
        train_input_ids, test_input_ids, train_sections, test_sections, train_labels, test_labels = splits

        fname = os.path.join(output_dir, 'test.tfrecord')
        write_examples(fname, test_input_ids, test_sections, test_labels)

        input_ids = train_input_ids
        sections = train_sections
        labels = train_labels
    elif args.test_size and args.test_size == 1.0:
        fname = os.path.join(output_dir, 'eval.tfrecord')
        write_examples(fname, input_ids, sections, labels)
        return

    fold = StratifiedKFold(n_splits=args.num_k_fold, shuffle=True, random_state=args.seed)
    for index, (train_indices, valid_indices) in enumerate(fold.split(input_ids, labels)):
        fname = os.path.join(
            output_dir, 'train.fold-{}-of-{}.tfrecord'.format(index + 1, args.num_k_fold)
        )
        write_examples(fname, input_ids, sections, labels, indices=train_indices)

        fname = os.path.join(
            output_dir, 'valid.fold-{}-of-{}.tfrecord'.format(index + 1, args.num_k_fold)
        )
        write_examples(fname, input_ids, sections, labels, indices=valid_indices)


def main():
    setup_formatter(logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--filepath', default='./data/tagging_train.json',
                        help='Location of data file')
    parser.add_argument('--model_name', default='klue/roberta-base',
                        help='Name of pre-trained models (korscibert, korscielectra, huggingface models)')
    parser.add_argument('--output_dir', default='./data/tfrecords',
                        help='Where to write out the tfrecords')
    parser.add_argument('--korscibert_vocab', default='./cort/pretrained/korscibert/vocab_kisti.txt',
                        help='Location of KorSci-BERT vocabulary file')
    parser.add_argument('--korscielectra_vocab', default='./cort/pretrained/korscielectra/data/vocab.txt',
                        help='Location of KorSci-ELECTRA vocabulary file')
    parser.add_argument('--num_processes', default=4, type=int,
                        help='Parallelize across multiple processes')
    parser.add_argument('--num_k_fold', default=10, type=int,
                        help='Number of K-Fold splits')
    parser.add_argument('--test_size', default=0, type=float,
                        help='Rate of testing dataset')
    parser.add_argument('--seed', default=42, type=int,
                        help='The seed of random state')

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name.replace('/', '_'))
    if os.path.exists(output_dir):
        logging.info('Cleaning up legacy directory')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Tokenizer: {}'.format(args.model_name))
    create_tfrecords(args, output_dir)
    logging.info('All jobs have finished')


if __name__ == '__main__':
    main()
