import json
import logging
import pandas as pd

from tqdm import tqdm


SECTION_NAMES = ['연구 목적', '연구 방법', '연구 결과']
LABEL_NAMES = ['문제 정의', '가설 설정', '기술 정의',
               '제안 방법', '대상 데이터', '데이터처리', '이론/모형',
               '성능/효과', '후속연구']


def parse_and_preprocess_sentences(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        data = json.load(f)

    dicts = {
        'sentences': [],
        'sections': [],
        'labels': []
    }
    logging.debug('Reading available rows...')
    for row in tqdm(data):
        label = row['tag']
        dicts['sentences'].append(row['sentence'])
        dicts['labels'].append(label)
        if label in ['문제 정의', '가설 설정', '기술 정의']:
            dicts['sections'].append('연구 목적')
        elif label in ['제안 방법', '대상 데이터', '데이터처리', '이론/모형']:
            dicts['sections'].append('연구 방법')
        elif label in ['성능/효과', '후속연구']:
            dicts['sections'].append('연구 결과')
        else:
            raise ValueError('Invalid label: {}'.format(label))

    df = pd.DataFrame(dicts)
    df['code_labels'] = df['labels'].apply(lambda name: LABEL_NAMES.index(name))
    df['code_sections'] = df['sections'].apply(lambda name: SECTION_NAMES.index(name))

    def print_description(name, stats):
        logging.debug('{}:'.format(name))
        titles = stats.index
        for title, counts in zip(titles, stats):
            logging.debug('- {}: {:,}'.format(title, counts))

    print_description('Section', df['sections'].value_counts())
    print_description('Labels', df['labels'].value_counts())

    return df
