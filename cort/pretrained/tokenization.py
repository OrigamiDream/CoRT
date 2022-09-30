from tqdm import tqdm
from typing import Union, List, Optional


class TokenizerDelegate:

    def __init__(self, delegate, max_length: int = 512):
        self.delegate = delegate
        self.max_length = max_length
        self.cls_token_id = self.delegate.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token_id = self.delegate.convert_tokens_to_ids(['[SEP]'])[0]
        self.pad_token_id = self.delegate.convert_tokens_to_ids(['[PAD]'])[0]
        self.num_reserved_tokens = 2

    def tokenize(self, text):
        return self.delegate.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return self.delegate.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.delegate.convert_ids_to_tokens(ids)

    def __call__(self,
                 texts: Union[str, List[str]],
                 padding: Optional[str] = None,
                 truncation=False,
                 return_attention_mask=True,
                 return_token_type_ids=True):
        if isinstance(texts, str):
            texts = [texts]

        dicts = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }
        for text in tqdm(texts, desc='Tokenizing', ncols=120):
            # tokenize to token_ids
            tokens = self.tokenize(text)
            tokens = self.convert_tokens_to_ids(tokens)

            # truncate
            if truncation:
                tokens = tokens[:min(len(tokens), self.max_length - self.num_reserved_tokens)]

            tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
            attention_mask = [1] * len(tokens)

            if padding is not None and padding == 'max_length':
                remains = self.max_length - len(tokens)
                pads = [self.pad_token_id] * remains
                tokens = tokens + pads
                attention_mask = [0] * remains
                assert len(tokens) == self.max_length, (
                    'Padded texts length must be {}, but received {} instead'
                    .format(self.max_length, len(tokens))
                )

            token_type_ids = [0] * len(tokens)

            dicts['input_ids'].append(tokens)
            dicts['attention_mask'].append(attention_mask)
            dicts['token_type_ids'].append(token_type_ids)

        return_value = {
            'input_ids': dicts['input_ids']
        }
        if return_attention_mask:
            return_value['attention_mask'] = dicts['attention_mask']
        if return_token_type_ids:
            return_value['token_type_ids'] = dicts['token_type_ids']
        return return_value


def create_tokenizer(vocab_file: str, tokenizer_type: str, do_lower_case=False, max_length=512):
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == 'electra':
        try:
            from cort.pretrained.korscielectra.model.tokenization import FullTokenizer as ElectraTokenizer
        except ImportError:
            raise ImportError('Failed to import KorSci-ELECTRA module.')
        tokenizer = ElectraTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    elif tokenizer_type == 'bert':
        try:
            from cort.pretrained.korscibert.tokenization_kisti import FullTokenizer as BertTokenizer
        except ImportError:
            raise ImportError('Failed to import KorSci-BERT module.')
        tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    else:
        raise AttributeError('Invalid tokenizer type: {}, Allowed: (electra, bert)'.format(tokenizer_type))

    return TokenizerDelegate(tokenizer, max_length=max_length)
