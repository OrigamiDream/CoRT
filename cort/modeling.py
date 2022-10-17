import logging
import tensorflow as tf

from utils import utils
from transformers import TFAutoModel
from cort.config import Config, ConfigLike
from cort.pretrained import migrator
from tensorflow.keras import models, layers, losses, initializers


def create_attention_mask(inputs, pad_token):
    return tf.cast(tf.math.not_equal(inputs, pad_token), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]


def create_token_type_ids(inputs):
    return tf.zeros_like(inputs, dtype=tf.int32)


def get_initializer(initializer_range=0.02):
    return initializers.TruncatedNormal(stddev=initializer_range)


def unwrap_inputs_with_class_weight(inputs, include_sections=False):
    labels = cw = None
    if len(inputs) == 1:
        input_ids = inputs
    elif len(inputs) == 2:
        input_ids, labels = inputs
    elif len(inputs) == 3:
        input_ids, labels, cw = inputs
    else:
        raise ValueError('Number of inputs must be 1, 2 and 3')

    if include_sections:
        labels = (
            (None, None) if labels is None else labels
        )
        cw = (
            (None, None) if cw is None else cw
        )

    return input_ids, labels, cw


def calc_hierarchical_contrastive_loss(pooled, sections, labels, temperature=0.07):
    depth = 2  # Number of category depth = 2 (sections, labels)

    hierarchy_masks = tf.ones((depth, depth), dtype=tf.float32)
    hierarchy_masks = tf.linalg.band_part(hierarchy_masks, 0, -1)  # upper triangular matrix
    hierarchy_masks = tf.stop_gradient(hierarchy_masks)

    cumulative_losses = []
    for level in range(depth):
        section_co_loss = calc_supervised_contrastive_loss(pooled, sections, temperature=temperature)
        label_co_loss = calc_supervised_contrastive_loss(pooled, labels, temperature=temperature)

        co_losses = tf.concat([section_co_loss, label_co_loss], axis=0)
        co_losses *= hierarchy_masks[level, :]
        co_loss = tf.reduce_mean(co_losses)

        # apply penalty to masked losses hierarchically
        penalty = tf.pow(2, tf.Variable(1 / (level + 1)))
        penalty = tf.stop_gradient(penalty)
        co_loss *= penalty

        cumulative_losses.append(co_loss)

    cumulative_losses = tf.concat(cumulative_losses, axis=0)
    return tf.reduce_mean(cumulative_losses)


def calc_margin_based_contrastive_loss(pooled, labels, eps=1e-3):
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

    max_dist = tf.reduce_max(dist * mask_pos)
    loss_pos = tf.reduce_sum(dist * mask_pos, axis=-1) / (tf.reduce_sum(mask_pos, axis=-1) + eps)
    loss_neg = tf.reduce_sum(tf.nn.relu(max_dist - dist) * mask_neg, axis=-1) / (tf.reduce_sum(mask_neg, axis=-1) + eps)
    cos_loss = tf.reduce_mean(loss_pos + loss_neg)
    return cos_loss


def calc_supervised_contrastive_loss(pooled, labels, temperature=0.3):
    # Supervised Contrastive Learning
    norm_pooled = tf.math.l2_normalize(pooled, axis=-1)
    cosine_score = tf.matmul(norm_pooled, norm_pooled, transpose_b=True) / temperature
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


def configure_backbone_model(config: Config):
    if config.model_name == 'korscielectra':
        logging.info('Migrating KorSci-ELECTRA')
        vocab = utils.parse_vocabulary(config.korscielectra_vocab)
        backbone = migrator.migrate_electra(config.korscielectra_ckpt, pad_token_id=vocab['[PAD]'])
    elif config.model_name == 'korscibert':
        logging.info('Migrating KorSci-BERT')
        vocab = utils.parse_vocabulary(config.korscibert_vocab)
        backbone = migrator.migrate_bert(config.korscibert_ckpt, pad_token_id=vocab['[PAD]'])
    else:
        logging.info('Loading `{}` from HuggingFace'.format(config.model_name))
        backbone = TFAutoModel.from_pretrained(config.model_name, from_pt=True)
    backbone.trainable = False

    if config.backbone_trainable_layers > 0:
        logging.info('Setting selective backbone encoder layers trainable:')

        encoder_layers = backbone.layers[0].encoder.layer
        num_trainable_layers = min(config.backbone_trainable_layers, len(encoder_layers))
        for i in reversed(range(len(encoder_layers) - num_trainable_layers, len(encoder_layers))):
            logging.info('- encoder/{} is now trainable'.format(encoder_layers[i].name))
            encoder_layers[i].trainable = True
    elif config.backbone_trainable_layers == -1:
        logging.info('Backbone model is trainable')
        backbone.trainable = True
    else:
        logging.info('Backbone model is completely frozen')
    return backbone


class CortForPretraining(models.Model):

    def __init__(self, config: ConfigLike, **kwargs):
        super(CortForPretraining, self).__init__(**kwargs)
        self.config = Config.parse_config(config)
        self.cort = CortMainLayer(config, name='cort', **kwargs)
        self.projection = CortRepresentationProjectionHead(config, name='projection', **kwargs)

    def call(self, inputs, training=None, mask=None):
        input_ids, (sections, labels) = inputs
        features = self.cort(input_ids, training=training)
        features = features.last_hidden_state[:, 0, :]
        representation = self.projection(features)

        if self.config.loss_base == 'margin':
            loss = calc_margin_based_contrastive_loss(representation, labels) * self.config.alpha
        elif self.config.loss_base == 'supervised':
            loss = calc_supervised_contrastive_loss(representation, labels)
        elif self.config.loss_base == 'hierarchical':
            loss = calc_hierarchical_contrastive_loss(representation, sections, labels)
        else:
            raise ValueError('Invalid contrastive loss base: {}'.format(self.config.loss_base))
        outputs = {
            'representation': representation,
            'labels': labels
        }
        return loss, outputs

    def get_config(self):
        return super(CortForPretraining, self).get_config()


class CortForElaboratedSequenceClassification(models.Model):

    def __init__(self, config: ConfigLike, num_sections: int, num_labels: int, **kwargs):
        super(CortForElaboratedSequenceClassification, self).__init__(**kwargs)
        assert config.train_in_once, (
            'CoRT for elaborated sequence classification requires to enable `train_in_once`'
        )

        self.config = Config.parse_config(config)
        self.num_sections = num_sections
        self.num_labels = num_labels

        self.cort = CortMainLayer(config, name='cort', **kwargs)
        if self.config.repr_classifier == 'seq_cls':
            self.section_classifier = CortClassificationHead(
                config, intermediate_size=self.config.repr_size, num_labels=num_sections, name='section_classifier'
            )
            self.label_classifier = CortClassificationHead(
                config, intermediate_size=self.config.repr_size, num_labels=num_labels, name='label_classifier'
            )
        elif self.config.repr_classifier == 'bi_lstm':
            self.section_classifier = CortBidirectionalClassificationHead(
                config, num_labels=num_sections, name='section_classifier'
            )
            self.label_classifier = CortBidirectionalClassificationHead(
                config, num_labels=num_labels, name='label_classifier'
            )
        else:
            raise ValueError('Invalid representation classifier: {}'.format(self.config.repr_classifier))
        self.loss_fn = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=losses.Reduction.NONE
        )

    def compute_losses(self, representation, logits, labels, cw):
        if labels is not None and self.config.loss_base == 'margin':
            co_loss = calc_margin_based_contrastive_loss(representation, labels)
        elif labels is not None and self.config.loss_base == 'supervised':
            co_loss = calc_supervised_contrastive_loss(representation, labels)
        elif labels is not None and self.config.loss_base == 'hierarchical':
            raise ValueError('Hierarchical contrastive loss is not supported in elaborated representation model')
        elif labels is None:
            co_loss = None
        else:
            raise ValueError('Invalid contrastive loss base: {}'.format(self.config.loss_base))

        cce_loss = None if labels is None else self.loss_fn(labels, logits, sample_weight=cw)
        return co_loss, cce_loss

    def call(self, inputs, training=None, mask=None):
        input_ids, (sections, labels), (section_cw, label_cw) = unwrap_inputs_with_class_weight(inputs)

        # One-Hot Encoding
        section_ohe_labels = (
            None if sections is None else tf.one_hot(sections, depth=self.num_sections, dtype=tf.float32)
        )
        ohe_labels = None if labels is None else tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)

        # Forward
        outputs = self.cort(input_ids, training=training)
        section_logits, section_representation = self.section_classifier(
            outputs, training=training, return_representation=True
        )
        logits, representation = self.label_classifier(
            outputs, training=training, return_representation=True, previous_representation=section_representation
        )

        section_probs = tf.cast(tf.nn.softmax(section_logits), dtype=tf.float32)
        probs = tf.cast(tf.nn.softmax(logits), dtype=tf.float32)

        cort_outputs = {
            'labels': labels,
            'logits': logits,
            'probs': probs,
            'ohe_labels': ohe_labels,
            'section_labels': sections,
            'section_logits': section_logits,
            'section_probs': section_probs,
            'section_ohe_labels': section_ohe_labels
        }

        # Compute Losses
        total_loss = None
        section_co_loss, section_cce_loss = self.compute_losses(
            section_representation, section_logits, sections, section_cw
        )
        co_loss, cce_loss = self.compute_losses(representation, logits, labels, label_cw)
        cort_outputs.update({
            'section_co_loss': section_co_loss,
            'section_cce_loss': section_cce_loss,
            'co_loss': co_loss,
            'cce_loss': cce_loss
        })
        if labels is not None:
            total_loss = (
                (section_cce_loss + (self.config.alpha * section_co_loss)) + (cce_loss + (self.config.alpha * co_loss))
            )
        return total_loss, cort_outputs

    def get_config(self):
        return super(CortForElaboratedSequenceClassification, self).get_config()


class CortForSequenceClassification(models.Model):

    def __init__(self, config: ConfigLike, num_labels: int, **kwargs):
        super(CortForSequenceClassification, self).__init__(**kwargs)
        self.config = Config.parse_config(config)
        self.num_labels = num_labels

        self.cort = CortMainLayer(config, name='cort', **kwargs)
        if self.config.repr_classifier == 'seq_cls':
            intermediate_size = (
                self.config.repr_size if self.config.train_in_once else self.config.pretrained_config.hidden_size
            )
            self.classifier = CortClassificationHead(
                config, intermediate_size=intermediate_size, num_labels=num_labels, name='classifier'
            )
        elif self.config.repr_classifier == 'bi_lstm':
            self.classifier = CortBidirectionalClassificationHead(config, num_labels=num_labels, name='classifier')
        else:
            raise ValueError('Invalid representation classifier: {}'.format(self.config.repr_classifier))
        self.loss_fn = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=losses.Reduction.NONE
        )

    def call(self, inputs, training=None, mask=None):
        input_ids, (sections, labels), (_, cw) = unwrap_inputs_with_class_weight(inputs)

        # Forward
        outputs = self.cort(input_ids, training=training)
        logits, representation = self.classifier(outputs, training=training, return_representation=True)

        probs = tf.cast(tf.nn.softmax(logits), dtype=tf.float32)
        ohe_labels = None if labels is None else tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)

        cort_outputs = {
            'logits': logits,
            'probs': probs,
            'labels': labels,
            'ohe_labels': ohe_labels
        }

        # Compute Losses
        cce_loss = None if labels is None else self.loss_fn(labels, logits, sample_weight=cw)

        total_loss = cce_loss
        if self.config.train_in_once:
            if labels is not None and self.config.loss_base == 'margin':
                co_loss = calc_margin_based_contrastive_loss(representation, labels)
            elif labels is not None and self.config.loss_base == 'supervised':
                co_loss = calc_supervised_contrastive_loss(representation, labels)
            elif labels is not None and self.config.loss_base == 'hierarchical':
                co_loss = calc_hierarchical_contrastive_loss(representation, sections, labels)
            elif labels is None:
                co_loss = None
            else:
                raise ValueError('Invalid contrastive loss base: {}'.format(self.config.loss_base))

            total_loss = None if labels is None else cce_loss + (self.config.alpha * co_loss)
            cort_outputs.update({
                'co_loss': co_loss,
                'cce_loss': cce_loss,
                'representation': representation
            })
        return total_loss, cort_outputs

    def get_config(self):
        return super(CortForSequenceClassification, self).get_config()


class CortMainLayer(layers.Layer):

    def __init__(self, config: ConfigLike, trainable=True, name=None, *args, **kwargs):
        super(CortMainLayer, self).__init__(trainable=trainable, name=name, *args, **kwargs)
        self.config = Config.parse_config(config)
        self.backbone = configure_backbone_model(self.config)

    def call(self, input_ids, *args, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else None
        attention_mask = tf.cast(tf.math.not_equal(input_ids, self.backbone.config.pad_token_id), dtype=tf.int32)
        token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

        hidden_state = self.backbone(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     output_hidden_states=True,
                                     training=training)
        return hidden_state

    def get_config(self):
        config = super(CortMainLayer, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config


class CortRepresentationProjectionHead(layers.Layer):

    def __init__(self, config: ConfigLike, trainable=True, name=None, *args, **kwargs):
        super(CortRepresentationProjectionHead, self).__init__(trainable=trainable, name=name, *args, **kwargs)
        self.config = Config.parse_config(config)
        self.dropout = layers.Dropout(self.config.classifier_dropout_prob, name='dropout')
        self.repr = layers.Dense(self.config.repr_size,
                                 kernel_initializer=get_initializer(self.config.pretrained_config.initializer_range),
                                 name='repr')
        self.activation = (
            layers.Activation(self.config.repr_act, name='{}_act'.format(self.config.repr_act))
            if self.config.repr_act != 'none' else None
        )

    def call(self, inputs, *args, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else None
        x = self.dropout(inputs, training=training)
        x = self.repr(x)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CortRepresentationProjectionHead, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config


class CortBidirectionalClassificationHead(layers.Layer):

    def __init__(self, config: ConfigLike, num_labels: int, **kwargs):
        super(CortBidirectionalClassificationHead, self).__init__(**kwargs)
        self.config = Config.parse_config(config)
        self.num_labels = num_labels

        self.concat_hidden = (
            layers.Concatenate(name='concat_hidden') if config.concat_hidden_states > 1 else None
        )
        self.bidirectional = layers.Bidirectional(layer=layers.LSTM(64, return_sequences=True), name='bi_lstm')
        self.average_pool = layers.GlobalAveragePooling1D(name='avg_pool')
        self.max_pool = layers.GlobalMaxPooling1D(name='max_pool')
        self.concatenate = layers.Concatenate(name='concatenate')
        self.dropout = layers.Dropout(self.config.classifier_dropout_prob, name='dropout')
        self.classifier = layers.Dense(
            self.num_labels,
            kernel_initializer=get_initializer(self.config.pretrained_config.initializer_range),
            name='classifier',
        )
        self.activation = (
            layers.Activation(self.config.repr_act, name='{}_act'.format(self.config.repr_act))
            if self.config.repr_act != 'none' else None
        )

    def call(self, inputs, *args, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else None
        return_representation = kwargs['return_representation'] if 'return_representation' in kwargs else None
        previous_representation = kwargs['previous_representation'] if 'previous_representation' in kwargs else None
        if self.concat_hidden is not None:
            hidden_states = inputs.hidden_states
            hidden_state = self.concat_hidden([
                hidden_states[(idx + 1) * -1] for idx in range(self.config.concat_hidden_states)
            ])
        else:
            hidden_state = inputs.last_hidden_state

        x = self.bidirectional(hidden_state, training=training)

        avg_pooled = self.average_pool(x, training=training)
        max_pooled = self.max_pool(x, training=training)

        x = self.concatenate([avg_pooled, max_pooled], training=training)
        x = self.dropout(x, training=training)

        # Representations
        if previous_representation is not None:
            x += previous_representation
        if self.config.repr_preact and self.activation is not None:
            x = self.activation(x)
        representation = x
        if not self.config.repr_preact and self.activation is not None:
            x = self.activation(x)

        x = self.classifier(x, training=training)
        if return_representation:
            return x, representation
        else:
            return x

    def get_config(self):
        config = super(CortBidirectionalClassificationHead, self).get_config()
        config.update({
            'config': self.config.to_dict(),
            'num_labels': self.num_labels
        })
        return config


class CortClassificationHead(layers.Layer):

    def __init__(self, config: ConfigLike, intermediate_size: int, num_labels: int, **kwargs):
        super(CortClassificationHead, self).__init__(**kwargs)
        self.config = Config.parse_config(config)
        self.intermediate_size = intermediate_size
        self.num_labels = num_labels

        self.concat_hidden = (
            layers.Concatenate(name='concat_hidden') if config.concat_hidden_states > 1 else None
        )
        self.dense = layers.Dense(
            intermediate_size,
            kernel_initializer=get_initializer(config.pretrained_config.initializer_range),
            name='dense'
        )
        self.activation = (
            layers.Activation(self.config.repr_act, name='{}_act'.format(self.config.repr_act))
            if self.config.repr_act != 'none' else None
        )
        self.dropout = layers.Dropout(config.classifier_dropout_prob, name='dropout')
        self.classifier = layers.Dense(
            num_labels,
            kernel_initializer=get_initializer(config.pretrained_config.initializer_range),
            name='classifier'
        )

    def call(self, inputs, *args, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else None
        return_representation = kwargs['return_representation'] if 'return_representation' in kwargs else None
        previous_representation = kwargs['previous_representation'] if 'previous_representation' in kwargs else None
        if self.concat_hidden is not None:
            hidden_states = inputs.hidden_states
            hidden_state = self.concat_hidden([
                hidden_states[(idx + 1) * -1] for idx in range(self.config.concat_hidden_states)
            ])
        else:
            hidden_state = inputs.last_hidden_state

        x = hidden_state[:, 0, :]
        x = self.dropout(x, training=training)
        x = self.dense(x)

        # Representations
        if previous_representation is not None:
            x += previous_representation
        if self.config.repr_preact and self.activation is not None:
            x = self.activation(x)
        representation = x
        if not self.config.repr_preact and self.activation is not None:
            x = self.activation(x)

        x = self.dropout(x, training=training)
        x = self.classifier(x)

        if return_representation:
            return x, representation
        else:
            return x

    def get_config(self):
        config = super(CortClassificationHead, self).get_config()
        config.update({
            'config': self.config.to_dict(),
            'intermediate_size': self.intermediate_size,
            'num_labels': self.num_labels
        })
        return config
