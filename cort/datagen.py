import numpy as np

from cort.config import Config, ConfigLike
from tensorflow.keras import utils


class CortDataGenerator(utils.Sequence):

    def __init__(self, config: ConfigLike, tokenizer,
                 sentences: np.ndarray,
                 sections: np.ndarray, labels: np.ndarray,
                 steps_per_epoch: int,
                 shuffle=True):
        self.config = Config.parse_config(config)
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.sections = sections
        self.labels = labels
        self.steps_per_epoch = steps_per_epoch
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed=config.seed)
        self.num_samples = len(self.sentences)
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()

    def __getitem__(self, index):
        to_index = (index + 1) * self.config.batch_size
        to_index = min(to_index, self.num_samples)
        indices = self.indices[index * self.config.batch_size:int(to_index)]

        sentences = list(self.sentences[indices])
        if self.shuffle:
            input_ids = []
            for sentence in sentences:
                words = sentence.split()
                limit = int(len(words) * self.config.word_mask_prob)
                word_masks = np.arange(len(words)) > limit
                self.random_state.shuffle(word_masks)
                filtered_words = [word for word, is_allowed in zip(words, word_masks) if is_allowed]
                input_ids.append(' '.join(filtered_words))
        else:
            input_ids = sentences

        tokenized = self.tokenizer(input_ids,
                                   padding='max_length',
                                   truncation=True,
                                   return_attention_mask=False,
                                   return_token_type_ids=False)
        input_ids = tokenized['input_ids']
        input_ids = np.array(input_ids, dtype=np.int32)

        sections = self.sections[indices]
        labels = self.labels[indices]
        return input_ids, (sections, labels)

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state.shuffle(self.indices)
