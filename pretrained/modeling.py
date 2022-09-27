import json

from pretrained.korscibert.modeling import BertModel, BertConfig


def create_bert():
    with open('./korscibert/bert_config_kisti.json') as f:
        body = json.load(f)

    # TODO: Migrate TensorFlow 1 model to TensorFlow 2
