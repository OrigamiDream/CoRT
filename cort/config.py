import copy

from typing import Dict, Any, Union


class Config:

    def __init__(self, **kwargs):
        self.pretrained_config = kwargs.pop('pretrained_config', {})
        self.train_path = kwargs.pop('train_path', './data/tagging_train.json')

        # Training Hyperparameters
        self.seed = kwargs.pop('seed', 42)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.epochs = kwargs.pop('epochs', 10000)
        self.initial_epoch = kwargs.pop('initial_epoch', 0)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.optimizer_clip_value = kwargs.pop('optimizer_clip_value', None)
        self.optimizer_clip_norm = kwargs.pop('optimizer_clip_norm', None)
        self.num_labels = kwargs.pop('num_labels', 9)
        self.num_sections = kwargs.pop('num_sections', 3)
        self.repr_size = kwargs.pop('repr_size', 1024)  # Size of dense representation layer for Contrastive Learning

        self.num_k_fold = kwargs.pop('num_f_fold', 10)
        self.current_fold = kwargs.pop('current_fold', 0)
        self.model_name = kwargs.pop('model_name', 'klue/roberta-base')

        # Model Hyperparameters
        self.classifier_dropout_prob = kwargs.pop('classifier_dropout_prob', 0.1)
        self.loss_base = kwargs.pop('loss_base', 'margin')
        self.alpha = kwargs.pop('alpha', 2.0)

        # Constants
        self.korscibert_ckpt = kwargs.pop('korscibert_ckpt',
                                          './cort/pretrained/korscibert/model.ckpt-262500')
        self.korscielectra_ckpt = kwargs.pop('korscielectra_ckpt',
                                             './cort/pretrained/korscielectra/data/models/korsci_base')

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        return cls(**config_dict)

    @staticmethod
    def parse_config(config: Union["Config", Dict[str, Any]]):
        if isinstance(config, dict):
            return Config(**config)
        return config


ConfigLike = Union[Config, Dict[str, Any]]
