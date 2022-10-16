import copy

from typing import Dict, Any, Union


class Config:

    def __init__(self, **kwargs):
        self.pretrained_config = kwargs.pop('pretrained_config', {})
        self.train_path = kwargs.pop('train_path', './data/tagging_train.json')

        # Training Hyperparameters
        self.seed = kwargs.pop('seed', 42)
        self.gpu = kwargs.pop('gpu', 'all')
        self.batch_size = kwargs.pop('batch_size', 64)
        self.distribute = kwargs.pop('distribute', False)
        self.epochs = kwargs.pop('epochs', 10)
        self.initial_epoch = kwargs.pop('initial_epoch', 0)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.lr_fn = kwargs.pop('lr_fn', 'cosine_decay')  # constant, cosine_decay, polynomial_decay, linear_decay
        self.lr_poly_decay_power = kwargs.pop('lr_poly_decay_power', 0.5)
        self.weight_decay = kwargs.pop('weight_decay', 1e-6)
        self.warmup_rate = kwargs.pop('warmup_rate', 0.06)
        self.cosine_annealing_freq = kwargs.pop('cosine_annealing_freq', 3000)
        self.layerwise_lr_decay = kwargs.pop('layerwise_lr_decay', 0.0)
        self.optimizer_clip_value = kwargs.pop('optimizer_clip_value', 0.5)
        self.optimizer_clip_norm = kwargs.pop('optimizer_clip_norm', 1.0)
        self.num_labels = kwargs.pop('num_labels', 9)
        self.num_sections = kwargs.pop('num_sections', 3)
        self.repr_size = kwargs.pop('repr_size', 1024)  # Size of dense representation layer for Contrastive Learning
        self.gradient_accumulation_steps = kwargs.pop('gradient_accumulation_steps', 1)
        self.skip_early_eval = kwargs.pop('skip_early_eval', False)
        self.word_mask_prob = kwargs.pop('word_mask_prob', 0.15)
        self.dynamic_datagen = kwargs.pop('dynamic_datagen', False)

        self.cross_validation = kwargs.pop('cross_validation', 'kfold')  # kfold, hyperparams
        self.num_k_fold = kwargs.pop('num_f_fold', 10)
        self.current_fold = kwargs.pop('current_fold', 0)
        self.model_name = kwargs.pop('model_name', 'klue/roberta-base')

        # Pre-training Hyperparameters
        self.num_train_steps = kwargs.pop('num_train_steps', 10000)
        self.save_checkpoint_steps = kwargs.pop('save_checkpoint_steps', 100)
        self.keep_checkpoint_max = kwargs.pop('keep_checkpoint_max', 3)
        self.restore_checkpoint = kwargs.pop('restore_checkpoint', '')
        self.log_freq = kwargs.pop('log_freq', 2000)

        # Model Hyperparameters
        self.repr_classifier = kwargs.pop('repr_classifier', 'seq_cls')  # seq_cls, bi_lstm
        self.repr_act = kwargs.pop('repr_act', 'tanh')
        self.classifier_dropout_prob = kwargs.pop('classifier_dropout_prob', 0.1)
        self.classifier_act = kwargs.pop('classifier_act', 'gelu')
        self.backbone_trainable_layers = kwargs.pop('backbone_trainable_layers', 0)
        self.concat_hidden_states = kwargs.pop('concat_hidden_states', 1)
        self.loss_base = kwargs.pop('loss_base', 'margin')
        self.include_sections = kwargs.pop('include_sections', False)
        self.repr_preact = kwargs.pop('repr_preact', True)
        self.alpha = kwargs.pop('alpha', 2.0)

        # Constants
        self.tfrecord_name = kwargs.pop('tfrecord_name',
                                        './data/tfrecords/{model_name}/{scope}.fold-{index}-of-{fold}.tfrecord')
        self.korscibert_vocab = kwargs.pop('korscibert_vocab', './cort/pretrained/korscibert/vocab_kisti.txt')
        self.korscibert_ckpt = kwargs.pop('korscibert_ckpt', './cort/pretrained/korscibert/model.ckpt-262500')
        self.korscielectra_vocab = kwargs.pop('korscielectra_vocab', './cort/pretrained/korscielectra/data/vocab.txt')
        self.korscielectra_ckpt = kwargs.pop('korscielectra_ckpt',
                                             './cort/pretrained/korscielectra/data/models/korsci_base')
        self.num_processes = kwargs.pop('num_processes', -1)

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
