import logging
import collections

import tensorflow as tf

from transformers import TFAutoModel
from cort.config import Config, ConfigLike
from cort.pretrained import migrator
from tensorflow.keras import models, layers, losses, initializers
from tensorflow.python.keras.utils import losses_utils


def create_attention_mask(inputs, pad_token):
    return tf.cast(tf.math.not_equal(inputs, pad_token), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]


def create_token_type_ids(inputs):
    return tf.zeros_like(inputs, dtype=tf.int32)


def get_initializer(initializer_range=0.02):
    return initializers.TruncatedNormal(stddev=initializer_range)


class CortModel(models.Model):

    def __init__(self, config: ConfigLike, **kwargs):
        super(CortModel, self).__init__(**kwargs)
        self.config = Config.parse_config(config)

        if config.model_name == 'korscielectra':
            logging.info('Migrating KorSci-ELECTRA')
            self.backbone = migrator.migrate_electra(config.korscielectra_ckpt)
        elif config.model_name == 'korscibert':
            logging.info('Migrating KorSci-BERT')
            self.backbone = migrator.migrate_bert(config.korscibert_ckpt)
        else:
            logging.info('Loading `{}` from HuggingFace'.format(config.model_name))
            self.backbone = TFAutoModel.from_pretrained(config.model_name, from_pt=True)
        self.backbone.trainable = False

        if config.backbone_trainable_layers > 0:
            logging.info('Setting selective backbone encoder layers trainable:')

            encoder_layers = self.backbone.layers[0].encoder.layer
            num_trainable_layers = min(config.backbone_trainable_layers, len(encoder_layers))
            for i in reversed(range(len(encoder_layers) - num_trainable_layers, len(encoder_layers))):
                logging.info('- encoder/{} is now trainable'.format(encoder_layers[i].name))
                encoder_layers[i].trainable = True
        elif config.backbone_trainable_layers == -1:
            logging.info('Backbone model is trainable')
            self.backbone.trainable = True
        else:
            logging.info('Backbone model is completely frozen')

        if self.config.repr_classifier == 'seq_cls':
            self.heading = CortForSequenceRepresentation(config, name='seq_repr')
        elif self.config.repr_classifier == 'bi_lstm':
            self.heading = CortForBidirectionalSequenceRepresentation(config, name='bi_seq_repr')
        else:
            raise ValueError('Invalid representation classification head: {}'.format(self.config.repr_classifier))

    def compute_backbone_representation(self, input_ids, training=True):
        attention_mask = tf.cast(tf.math.not_equal(input_ids, self.backbone.config.pad_token_id), dtype=tf.int32)
        token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

        hidden_state = self.backbone(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     training=training)
        representation, logits = self.heading(hidden_state.last_hidden_state, training=training)
        BackboneOutput = collections.namedtuple('BackboneOutput', [
            'attention_mask', 'token_type_ids', 'hidden_state', 'representation', 'logits'
        ])
        return BackboneOutput(attention_mask=attention_mask, token_type_ids=token_type_ids, hidden_state=hidden_state,
                              representation=representation, logits=logits)

    def calc_margin_based_contrastive_loss(self, pooled, labels):
        # Margin-based Contrastive Learning

        # ([B, 1, 1024] - [1, B, 1024])^2 = [B, B, 1024]
        dist = (tf.expand_dims(pooled, axis=1) - tf.expand_dims(pooled, axis=0)) ** 2
        # mean([B, B, 1024], axis=-1) = [B, B]
        dist = tf.reduce_mean(dist, axis=-1)

        # ([B, 1] == [1, B]) = [B, B]
        mask_pos = (tf.expand_dims(labels, axis=1) == tf.expand_dims(labels, axis=0))
        mask_pos = tf.cast(mask_pos, dtype=tf.float32)
        mask_pos = mask_pos - tf.linalg.diag(tf.linalg.diag_part(mask_pos))

        mask_neg = (tf.expand_dims(labels, axis=1) != tf.expand_dims(labels, axis=0))
        mask_neg = tf.cast(mask_neg, dtype=tf.float32)

        max_dist = tf.reduce_max(dist * mask_pos, axis=-1)
        loss_pos = tf.reduce_sum(dist * mask_pos, axis=-1) / (tf.reduce_sum(mask_pos, axis=-1) + 1e-3)
        loss_neg = tf.reduce_sum(tf.nn.relu(max_dist - dist) * mask_neg, axis=-1) / (tf.reduce_sum(mask_neg, axis=-1) + 1e-3)
        cos_loss = tf.reduce_mean(loss_pos + loss_neg)
        return cos_loss

    def calc_supervised_contrastive_loss(self, pooled, labels):
        # Supervised Contrastive Learning
        norm_pooled = tf.math.l2_normalize(pooled, axis=-1)
        cosine_score = tf.matmul(norm_pooled, norm_pooled, transpose_b=True) / 0.3
        cosine_score = tf.exp(cosine_score)
        cosine_score = cosine_score - tf.linalg.diag(tf.linalg.diag_part(cosine_score))

        mask_pos = (tf.expand_dims(labels, axis=1) == tf.expand_dims(labels, axis=0))
        mask_pos = tf.cast(mask_pos, dtype=tf.float32)
        mask_pos = mask_pos - tf.linalg.diag(tf.linalg.diag_part(mask_pos))

        cos_loss = cosine_score / tf.reduce_sum(cosine_score, axis=-1, keepdims=True)
        cos_loss = -tf.math.log(cos_loss + 1e-5)
        cos_loss = tf.reduce_sum(mask_pos * cos_loss, axis=-1) / (tf.reduce_sum(mask_pos, axis=-1) + 1e-3)
        cos_loss = tf.reduce_mean(cos_loss)
        return cos_loss

    def call(self, inputs, training=None, mask_pos=None):
        # input_ids shape must be [B, 512]
        # labels shape must be [B,], which is not one-hot encoded.
        if len(inputs) == 2:
            input_ids, labels = inputs
            class_weights = None
        elif len(inputs) == 3:
            input_ids, labels, class_weights = inputs
        else:
            raise ValueError(
                'Number of inputs must be 2 or 3. 2 for [input_ids, labels] and 3 for [input_ids, labels, class_weight]'
            )
        outputs = self.compute_backbone_representation(input_ids, training=training)

        # calculate losses
        pooled = outputs.representation
        logits = outputs.logits

        if self.config.loss_base == 'margin':
            cos_loss = self.calc_margin_based_contrastive_loss(pooled, labels)
        elif self.config.loss_base == 'supervised':
            cos_loss = self.calc_supervised_contrastive_loss(pooled, labels)
        else:
            raise ValueError('Invalid contrastive loss base: {}'.format(self.config.loss_base))

        ohe_labels = tf.one_hot(labels, depth=self.config.num_labels, dtype=tf.float32)
        cce_loss = tf.nn.softmax_cross_entropy_with_logits(ohe_labels, logits)
        cce_loss = losses_utils.compute_weighted_loss(losses=cce_loss, sample_weight=class_weights,
                                                      reduction=losses.Reduction.NONE)

        total_loss = cce_loss + (self.config.alpha * cos_loss)

        probs = tf.cast(tf.nn.softmax(logits), dtype=tf.float32)
        cort_outputs = {
            'probs': probs,
            'ohe_labels': ohe_labels,
            'contrastive_loss': cos_loss,
            'cross_entropy_loss': cce_loss,
        }
        return total_loss, cort_outputs

    def get_config(self):
        return super(CortModel, self).get_config()


class CortForBidirectionalSequenceRepresentation(layers.Layer):

    def __init__(self, config: ConfigLike, trainable=True, name=None, *args, **kwargs):
        super(CortForBidirectionalSequenceRepresentation, self).__init__(trainable=trainable,
                                                                         name=name, *args, **kwargs)
        self.config = Config.parse_config(config)
        self.bidirectional = None
        self.average_pool = self.max_pool = None
        self.concatenate = self.dropout = self.classifier = None

    def build(self, input_shape):
        self.bidirectional = layers.Bidirectional(layer=layers.LSTM(64, return_sequences=True), name='bi_lstm')
        self.average_pool = layers.GlobalAveragePooling1D(name='avg_pool')
        self.max_pool = layers.GlobalMaxPooling1D(name='max_pool')
        self.concatenate = layers.Concatenate(name='concatenate')
        self.dropout = layers.Dropout(self.config.classifier_dropout_prob, name='dropout')

        initializer_range = self.config.pretrained_config.initializer_range
        self.classifier = layers.Dense(self.config.num_labels, name='classifier',
                                       kernel_initializer=get_initializer(initializer_range))
        self.built = True

    def call(self, inputs, *args, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else None
        x = self.bidirectional(inputs, training=training)

        avg_pooled = self.average_pool(x, training=training)
        max_pooled = self.max_pool(x, training=training)

        representation = self.concatenate([avg_pooled, max_pooled], training=training)

        x = self.dropout(representation, training=training)
        logits = self.classifier(x, training=training)
        return representation, logits

    def get_config(self):
        config = super(CortForBidirectionalSequenceRepresentation, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config


class CortForSequenceRepresentation(layers.Layer):

    def __init__(self, config: ConfigLike, trainable=True, name=None, *args, **kwargs):
        super(CortForSequenceRepresentation, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        self.config = Config.parse_config(config)
        self.repr = None
        self.dropout = None
        self.classifier = None
        self.activation = None

    def build(self, input_shape):
        initializer_range = self.config.pretrained_config.initializer_range
        self.repr = layers.Dense(self.config.repr_size, name='repr',
                                 kernel_initializer=get_initializer(initializer_range))
        self.dropout = layers.Dropout(self.config.classifier_dropout_prob, name='dropout')
        self.classifier = layers.Dense(self.config.num_labels, name='classifier',
                                       kernel_initializer=get_initializer(initializer_range))
        if self.config.repr_act != 'none':
            self.activation = layers.Activation(self.config.repr_act)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else None

        # [CLS] token embedding
        hidden_state = inputs[:, 0, :]  # transformer.last_hidden_state[:, 0, :]
        x = self.dropout(hidden_state, training=training)
        x = self.repr(x, training=training)

        if self.activation is not None:
            representation = self.activation(x)
        else:
            representation = x

        x = self.dropout(representation, training=training)
        logits = self.classifier(x, training=training)

        return representation, logits

    def get_config(self):
        config = super(CortForSequenceRepresentation, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config
