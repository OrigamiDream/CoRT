import copy

from typing import Dict, Any, Union


class Config:

    def __init__(self, **kwargs):
        self.pretrained_config = kwargs.pop('pretrained_config', {})

        # Environmental Hyperparameters
        self.num_labels = kwargs.pop('num_labels', 9)
        self.num_sections = kwargs.pop('num_sections', 3)

        # Location of TFRecords and those naming formats.
        # `model_name` is one of `korscibert`, `korscielectra`, and pre-trained huggingface transformers,
        # `scope` is one of `train` and `valid`, `index` is the index of K-Fold,
        # and `fold` is the total number of folds.
        self.tfrecord_name = kwargs.pop('tfrecord_name',
                                        './data/tfrecords/{model_name}/{scope}.fold-{index}-of-{fold}.tfrecord')
        self.korscibert_vocab = kwargs.pop('korscibert_vocab', './cort/pretrained/korscibert/vocab_kisti.txt')
        self.korscibert_ckpt = kwargs.pop('korscibert_ckpt', './cort/pretrained/korscibert/model.ckpt-262500')
        self.korscielectra_vocab = kwargs.pop('korscielectra_vocab', './cort/pretrained/korscielectra/data/vocab.txt')
        self.korscielectra_ckpt = kwargs.pop('korscielectra_ckpt',
                                             './cort/pretrained/korscielectra/data/models/korsci_base')

        # Location of Pre-trained checkpoint directory and its naming format. `run_name` is W&B Run ID
        self.pretraining_checkpoint_dir = kwargs.pop('pretraining_checkpoint_dir',
                                                     './pretraining-checkpoints/{run_name}')
        self.num_processes = kwargs.pop('num_processes', -1)

        # Training Hyperparameters
        self.seed = kwargs.pop('seed', 42)

        # Restricting GPU allocation. `all` is none of allocation control from this application.
        # Otherwise, the number of GPU device is allowed. e.g., 0 and 1
        self.gpu = kwargs.pop('gpu', 'all')
        self.batch_size = kwargs.pop('batch_size', 64)
        self.distribute = kwargs.pop('distribute', False)

        # Total number of train steps for Pre-training.
        self.num_train_steps = kwargs.pop('num_train_steps', 10000)

        # Total number of epochs for Fine-tuning
        # The total number of train steps is `epochs` * `steps_per_epoch`
        self.epochs = kwargs.pop('epochs', 10)
        self.initial_epoch = kwargs.pop('initial_epoch', 0)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)

        # Function of learning rate scheduler
        # one of `constant`, `cosine_decay`, `polynomial_decay` and `linear_decay` is allowed.
        self.lr_fn = kwargs.pop('lr_fn', 'cosine_decay')

        # The decaying rate in case of learning rate scheduler is `polynomial_decay`
        self.lr_poly_decay_power = kwargs.pop('lr_poly_decay_power', 0.5)
        self.weight_decay = kwargs.pop('weight_decay', 1e-6)

        # Step rate for linear warming up of learning rate at beginning.
        # The total number of steps of warm up is `warmup_rate` * `total_train_steps`
        self.warmup_rate = kwargs.pop('warmup_rate', 0.06)
        self.cosine_annealing_freq = kwargs.pop('cosine_annealing_freq', 3000)
        self.layerwise_lr_decay = kwargs.pop('layerwise_lr_decay', 0.0)
        self.optimizer_clip_norm = kwargs.pop('optimizer_clip_norm', 1.0)

        # Size of dense representation layer for Contrastive Representation Learning
        self.repr_size = kwargs.pop('repr_size', 1024)
        self.gradient_accumulation_steps = kwargs.pop('gradient_accumulation_steps', 1)

        # On Fine-tuning stage, whether to train the model from scratch or not.
        self.train_at_once = kwargs.pop('train_at_once', False)

        # On Fine-tuning stage, whether to train the model with Contrastive Representation Learning
        # and classification task at the same time.
        self.repr_finetune = kwargs.pop('repr_finetune', False)
        self.num_k_fold = kwargs.pop('num_f_fold', 10)
        self.current_fold = kwargs.pop('current_fold', 0)
        self.save_checkpoint_steps = kwargs.pop('save_checkpoint_steps', 100)
        self.keep_checkpoint_max = kwargs.pop('keep_checkpoint_max', 5)
        self.restore_checkpoint = kwargs.pop('restore_checkpoint', '')
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', '')
        self.pretraining_run_name = kwargs.pop('pretraining_run_name', '')
        self.log_freq = kwargs.pop('log_freq', 2000)

        # Model Hyperparameters

        # Name of Pre-trained backbone model. One of `korscibert`, `korscielectra`
        # and identifier of Pre-trained hugging face transformers is allowed
        self.model_name = kwargs.pop('model_name', 'klue/roberta-base')

        # Architecture type of classification head in Fine-tuning stage.
        # One of `seq_cls` and `bi_lstm` is allowed.
        self.repr_classifier = kwargs.pop('repr_classifier', 'seq_cls')
        self.repr_act = kwargs.pop('repr_act', 'tanh')
        self.classifier_dropout_prob = kwargs.pop('classifier_dropout_prob', 0.1)
        self.backbone_trainable_layers = kwargs.pop('backbone_trainable_layers', 0)
        self.concat_hidden_states = kwargs.pop('concat_hidden_states', 1)

        # Type of loss function for Contrastive Representation Learning.
        # One of `margin`, `supervised` and `hierarchical` is allowed.
        self.loss_base = kwargs.pop('loss_base', 'margin')

        # One Fine-tuning stage, whether to train the classification model with section predictions additionally.
        self.include_sections = kwargs.pop('include_sections', False)
        self.repr_preact = kwargs.pop('repr_preact', True)
        self.alpha = kwargs.pop('alpha', 2.0)

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
