import re
import time
import json
import logging
import multiprocessing
import numpy as np
import pandas as pd

from tqdm import tqdm
from soynlp import normalizer
from typing import Union, Dict, Any, Iterable
from multiprocessing import Process, Queue


CHARACTER_FILTER_PATTERN = re.compile(r'[^ .,?!/@$%~％·∼()\x00-\x7F가-힣]+')
URL_FILTER_PATTERN = re.compile(
    r'https?://(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
)
EMAIL_FILTER_PATTERN = re.compile(r'[-\w.]+@([\w-]+\.)+[\w-]{2,4}')
TAGS_FILTER_PATTERN = re.compile(r'([#@]\S+)')
PERIOD_CORRECTION_PATTERN = re.compile(r'(( )(\. |\.\. ))')
PERIOD_REPEAT_PATTERN = re.compile(r'(\.)\1{2,}')
SPECIAL_CHAR_REPLACEMENTS = [
    ('χ', 'x'), ('ℓ', 'L'), ('β', '베타'), ('&lt;', '<'), ('&gt;', '>'), ('–', '-'),
    ('℃', '도씨'), ('α', '알파'), ('㎏', 'kg'), ('㎝', 'cm'), ('㎛', 'mm'), ('～', '~'),
    ('‧', '·'), ('ㆍ', '·'), ('：', ':'), ('°', '도'), ('×', 'x'), ('μ', 'm'), ('㏊', 'ha'),
    ('㎚', 'nm'), ('㎜', 'mm'), ('㎞', 'km'), ('㎟', 'mm'), ('㎠', 'cm'), ('㎡', 'm'),
    ('㎢', 'km'), ('㎕', 'mL'), ('㎖', 'mL'), ('㎗', 'dL'), ('㎣', 'mm'), ('㎤', 'cm'),
    ('㎥', 'm'), ('㎦', 'km'), ('㏄', 'cc'), ('팔꿉', '팔꿈치'), ('θ', '세타'), ('뻣나무', '벚나무'),
]

RESEARCH_PURPOSE = ['문제 정의', '가설 설정', '기술 정의']
RESEARCH_METHOD = ['제안 방법', '대상 데이터', '데이터처리', '이론/모형']
RESEARCH_RESULT = ['성능/효과', '후속연구']
SECTION_NAMES = ['연구 목적', '연구 방법', '연구 결과']
LABEL_NAMES = RESEARCH_PURPOSE + RESEARCH_METHOD + RESEARCH_RESULT


def parse_and_preprocess_sentences(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        data = json.load(f)

    dicts = {
        'sentences': [],
        'sections': [],
        'labels': []
    }
    logging.info('Reading available rows...')
    for row in tqdm(data):
        label = row['tag']
        dicts['sentences'].append(row['sentence'])
        dicts['labels'].append(label)
        if label in RESEARCH_PURPOSE:
            dicts['sections'].append('연구 목적')
        elif label in RESEARCH_METHOD:
            dicts['sections'].append('연구 방법')
        elif label in RESEARCH_RESULT:
            dicts['sections'].append('연구 결과')
        else:
            raise ValueError('Invalid label: {}'.format(label))

    df = pd.DataFrame(dicts)
    df['code_labels'] = df['labels'].apply(lambda name: LABEL_NAMES.index(name))
    df['code_sections'] = df['sections'].apply(lambda name: SECTION_NAMES.index(name))

    def print_description(name, stats):
        logging.info('{}:'.format(name))
        titles = stats.index
        for title, counts in zip(titles, stats):
            logging.info('- {}: {:,}'.format(title, counts))

    print_description('Section', df['sections'].value_counts())
    print_description('Labels', df['labels'].value_counts())

    return df


def current_milliseconds():
    return round(time.time() * 1000)


def format_minutes_and_seconds(milliseconds):
    minutes = int(milliseconds / 1000 / 60)
    seconds = int(milliseconds / 1000) - (minutes * 60)
    return minutes, seconds


def normalize_texts(sentence,
                    filter_specials=True,
                    filter_characters=True,
                    filter_urls=True,
                    filter_tags=True,
                    filter_emails=True,
                    period_correction=True,
                    concat_quotes=True,
                    normalize_repeats=True,
                    remove_spaces=True):
    if filter_specials:
        for before, after in SPECIAL_CHAR_REPLACEMENTS:
            sentence = sentence.replace(before, after)
    if filter_characters:
        sentence = CHARACTER_FILTER_PATTERN.sub(' ', sentence)
    if filter_urls:
        sentence = URL_FILTER_PATTERN.sub(' ', sentence)
    if filter_tags:
        sentence = TAGS_FILTER_PATTERN.sub(' ', sentence)
    if filter_emails:
        sentence = EMAIL_FILTER_PATTERN.sub(' ', sentence)
    if period_correction:
        sentence = PERIOD_REPEAT_PATTERN.sub(r'\1\1', sentence)
        sentence = PERIOD_CORRECTION_PATTERN.sub(r'\3', sentence)
    if concat_quotes:
        sentence = sentence.replace('``', '"').replace('\'\'', '"')
    if normalize_repeats:
        sentence = normalizer.repeat_normalize(sentence, num_repeats=2)
    if remove_spaces:
        sentence = ' '.join(sentence.strip().split())
    return sentence


def run_wrapped_job(q, index, fn, args=()):
    result = fn(*args)
    q.put((index, result))


def run_multiprocessing_job(fn,
                            data: Union[np.ndarray, Iterable[np.ndarray], Dict[Any, np.ndarray]],
                            num_processes=7, args=()):
    """
    Example:

    def counter(batch, multiplier):
        return batch * multiplier

    if __name__ == '__main__':
        data = np.arange(0, 100)
        multiplier = 10

        results = run_multiprocessing_job(counter, data, num_processes=5, args=(multiplier,))
        results = np.concatenate(results)
    """
    start_time = current_milliseconds()

    if isinstance(data, list) or isinstance(data, tuple):
        size = sum([len(element) for element in data])
        assert len(data) > 0, 'Length of data must be positive'

        data_size = len(data[0])
        assert size / len(data) == data_size, 'All data size must be same'
    elif isinstance(data, dict):
        size = sum([len(element) for element in data.values()])
        assert len(data.keys()) > 0, 'Length of data must be positive'

        data_size = len(list(data.values())[0])
        assert size / len(data.keys()) == data_size, 'All data size must be same'
    else:
        data_size = len(data)
        assert data_size > 0, 'Length of data must be positive'

    if num_processes == -1:
        num_processes = multiprocessing.cpu_count()
        logging.info('Auto-detecting number of available cores: {}'.format(num_processes))

    chunk_size = data_size // num_processes

    q = Queue()
    workers = []
    for i in range(num_processes):
        from_index = i * chunk_size
        to_index = data_size if i == num_processes - 1 else (i + 1) * chunk_size

        if isinstance(data, list) or isinstance(data, tuple):
            batch = [element[from_index:to_index] for element in data]
        elif isinstance(data, dict):
            batch = {key: element[from_index:to_index] for key, element in data.items()}
        else:
            batch = data[from_index:to_index]

        workers.append(Process(target=run_wrapped_job, args=(q, i, fn, (batch, *args))))
    logging.info('Starting {} multiprocessing workers...'.format(num_processes))
    [worker.start() for worker in workers]
    logging.info('{} workers have been started'.format(len(workers)))

    sorting_map = {}
    for i in range(len(workers)):
        index, result = q.get()
        sorting_map[index] = result

    values = []
    for i in range(len(workers)):
        values.append(sorting_map[i])

    logging.info('Finishing {} multiprocessing workers...'.format(len(workers)))
    [worker.join() for worker in workers]
    [worker.terminate() for worker in workers]

    time_elapsed = current_milliseconds() - start_time
    minutes, seconds = format_minutes_and_seconds(time_elapsed)

    logging.info('All workers have finished their jobs (time elapsed: {:02d}:{:02d})'.format(minutes, seconds))
    return values
