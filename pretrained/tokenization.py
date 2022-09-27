from pretrained.korscibert.tokenization_kisti import FullTokenizer as BertTokenizer
from pretrained.korscielectra.model.tokenization import FullTokenizer as ElectraTokenizer


def create_tokenizer(vocab_file: str, tokenizer_type: str, do_lower_case=False):
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == 'electra':
        tokenizer = ElectraTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    elif tokenizer_type == 'bert':
        tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    else:
        raise AttributeError('Invalid tokenizer type: {}, Allowed: (electra, bert)'.format(tokenizer_type))

    return tokenizer
