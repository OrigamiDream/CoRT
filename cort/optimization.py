import re
import collections
import tensorflow as tf

from cort.config import ConfigLike
from tensorflow.keras import optimizers
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.training import training_ops
from typing import Union, Callable


def create_optimizer(config: ConfigLike, total_train_steps, name=None):
    def _get_layer_decay(decay_rate, num_layers):
        key_to_depths = collections.OrderedDict({
            '/embedding/': 0,
            '/embeddings/': 0,
            '/embeddings_project/': 0
        })
        total_depth = 0
        for layer in range(num_layers):
            total_depth += 1
            key_to_depths['/layer_._{}/'.format(layer)] = total_depth

        head_layers = [
            # For CoRT Pretraining
            '/projection/',

            # For CoRT sequence classification
            '/classifier',

            # For CoRT elaborated sequence classification
            '/section_classifier/', '/label_classifier/',

            # For representation model heads
            '/seq_repr/', '/bi_seq_repr/',

            # For elaborated representation model heads
            '/sec_repr/', '/bi_sec_repr/'
        ]
        for layer in head_layers:
            key_to_depths[layer] = total_depth + 1

        return {
            key: decay_rate ** (total_depth + 1 - depth)
            for key, depth in key_to_depths.items()
        }

    warmup_apical_steps = int(max(1, total_train_steps * config.warmup_rate))

    if config.lr_fn == 'cosine_decay':
        learning_rate_fn = optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=config.learning_rate,
            first_decay_steps=config.cosine_annealing_freq,
            t_mul=1.0, m_mul=1.0
        )
    elif config.lr_fn == 'polynomial_decay':
        learning_rate_fn = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=total_train_steps - warmup_apical_steps,
            end_learning_rate=0.0,
            power=config.lr_poly_decay_power
        )
    elif config.lr_fn == 'linear_decay':
        learning_rate_fn = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=total_train_steps - warmup_apical_steps,
            end_learning_rate=0.0,
            power=1.0
        )
    else:
        raise ValueError('Invalid learning rate function type:', config.lr_fn)

    if config.warmup_rate:
        learning_rate_fn = LinearWarmUp(
            initial_learning_rate=config.learning_rate,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=warmup_apical_steps
        )

    layer_decay = None
    if config.layerwise_lr_decay:
        layer_decay = _get_layer_decay(config.layerwise_lr_decay, config.pretrained_config.num_hidden_layers)

    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=config.weight_decay,
        layer_decay=layer_decay,
        epsilon=1e-6,
        exclude_from_weight_decay=['layer_norm', 'bias', 'LayerNorm'],
        clip_norm=config.optimizer_clip_norm,
        name=name
    )
    return optimizer, learning_rate_fn


class LinearWarmUp(optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, epsilon=1e-12, name=None):
        super(LinearWarmUp, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps
        self.power = power
        self.epsilon = epsilon
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or 'LinearWarmUp') as name:
            epochf = tf.cast(step, dtype=tf.float32)
            warmup_stepsf = tf.cast(self.warmup_steps, dtype=tf.float32)
            warmup_progress = epochf / warmup_stepsf

            warmup_lr = self.initial_learning_rate * tf.math.pow(warmup_progress, self.power)
            warmup_lr += self.epsilon

            side_mask = tf.cast(step, dtype=tf.int64) <= self.warmup_steps
            lhs = tf.cast(side_mask, dtype=tf.float32)
            rhs = tf.cast(~side_mask, dtype=tf.float32)

            lhs *= warmup_lr
            rhs *= self.decay_schedule_fn(step - self.warmup_steps)

            return tf.add(lhs, rhs, name=name)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_schedule_fn': self.decay_schedule_fn,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
            'epsilon': self.epsilon,
            'name': self.name
        }


class AdamWeightDecay(optimizers.Adam):

    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients.
    Just adding the square of the weights to the loss function is *not* the
    correct way of using L2 regularization/weight decay with Adam, since that will
    interact with the m and v parameters in strange ways.
    Instead, we want ot decay the weights in a manner that doesn't interact with
    the m/v parameters. This is equivalent to adding the square of the weights to
    the loss with plain (non-momentum) SGD.
    """

    def __init__(
            self,
            learning_rate: Union[Callable, float] = 0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
            weight_decay_rate=0.0,
            include_in_weight_decay=None,
            exclude_from_weight_decay=None,
            layer_decay=None,
            clip_norm=1.0,
            name="AdamWeightDecay",
            **kwargs
    ):
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay
        self.layer_decay = layer_decay
        self.clip_norm = clip_norm

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {"LinearWarmUp": LinearWarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state["weight_decay_rate"] = tf.constant(self.weight_decay_rate, name="adam_weight_decay_rate")

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state["weight_decay_rate"], use_locking=self._use_locking
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        grads, tvars = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(
            zip(grads, tvars),
            name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients
        )

    def _get_lr(self, var, apply_state):
        """Retrieves the learning rate with the given state."""
        # if apply_state is None:
        #     return self._decayed_lr_t[var_dtype], {}
        var_name, var_device, var_dtype = var.name, var.device, var.dtype.base_dtype

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients
        lr_t = coefficients["lr_t"]
        lr = coefficients["lr"]

        if self.layer_decay is not None:
            update_for_var = False
            for key in self.layer_decay:
                if key in var_name:
                    update_for_var = True
                    lr_t *= self.layer_decay[key]
                    lr *= self.layer_decay[key]
                    break
            if not update_for_var:
                raise ValueError("No learning rate specified for variable", var)

        return lr_t, lr, coefficients, dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # print("Dense: {} {} {}".format(var.name, var.device, var.dtype.base_dtype))
        lr_t, _, coefficients, kwargs = self._get_lr(var, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            m = self.get_slot(var, 'm')
            v = self.get_slot(var, 'v')

            if not self.amsgrad:
                return training_ops.resource_apply_adam(
                    var.handle,
                    m.handle,
                    v.handle,
                    coefficients['beta_1_power'],
                    coefficients['beta_2_power'],
                    lr_t,
                    coefficients['beta_1_t'],
                    coefficients['beta_2_t'],
                    coefficients['epsilon'],
                    grad,
                    use_locking=self._use_locking)
            else:
                vhat = self.get_slot(var, 'vhat')
                return training_ops.resource_apply_adam_with_amsgrad(
                    var.handle,
                    m.handle,
                    v.handle,
                    vhat.handle,
                    coefficients['beta_1_power'],
                    coefficients['beta_2_power'],
                    lr_t,
                    coefficients['beta_1_t'],
                    coefficients['beta_2_t'],
                    coefficients['epsilon'],
                    grad,
                    use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # print("Sparse: {} {} {}".format(var.name, var.device, var.dtype.base_dtype))
        lr_t, lr, coefficients, kwargs = self._get_lr(var, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            # m_t = beta1 * m + (1 - beta1) * g_t
            m = self.get_slot(var, 'm')
            m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
            m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                                   use_locking=self._use_locking)
            with tf.control_dependencies([m_t]):
                m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v = self.get_slot(var, 'v')
            v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
            v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                                   use_locking=self._use_locking)
            with tf.control_dependencies([v_t]):
                v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

            if not self.amsgrad:
                v_sqrt = math_ops.sqrt(v_t)
                var_update = state_ops.assign_sub(
                    var, lr * m_t / (v_sqrt + coefficients['epsilon']),
                    use_locking=self._use_locking)
                return control_flow_ops.group(*[var_update, m_t, v_t])
            else:
                v_hat = self.get_slot(var, 'vhat')
                v_hat_t = math_ops.maximum(v_hat, v_t)
                with tf.control_dependencies([v_hat_t]):
                    v_hat_t = state_ops.assign(
                        v_hat, v_hat_t, use_locking=self._use_locking)
                v_hat_sqrt = math_ops.sqrt(v_hat_t)
                var_update = state_ops.assign_sub(
                    var,
                    lr * m_t / (v_hat_sqrt + coefficients['epsilon']),
                    use_locking=self._use_locking)
                return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


# Inspired from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py
class GradientAccumulator(object):
    """Distribution strategies-aware gradient accumulation utility."""

    def __init__(self):
        """Initializes the accumulator."""
        self.gradients = []
        self.accum_steps = tf.Variable(
            initial_value=0, dtype=tf.int32, trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

    @property
    def step(self):
        return self.accum_steps.value()

    @property
    def accumulated_gradients(self):
        return list(gradient.value() if gradient is not None else gradient for gradient in self.get_replica_gradients())

    def __call__(self, gradients):
        """Accumulates :obj:`gradients`."""
        if not self.gradients:
            self.gradients.extend([
                tf.Variable(tf.zeros_like(gradient), trainable=False) if gradient is not None else gradient
                for gradient in gradients
            ])

        if len(gradients) != len(self.gradients):
            raise ValueError('Expected %s gradients, but got %d' % (len(self.gradients), len(gradients)))

        for accum_gradient, gradient in zip(self.get_replica_gradients(), gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient)

        self.accum_steps.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients."""
        if self.gradients:
            self.accum_steps.assign(0)

        for gradient in self.get_replica_gradients():
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))

    def get_replica_gradients(self):
        if tf.distribute.has_strategy():
            replica_context = tf.distribute.get_replica_context()
            if replica_context is None or tf.distribute.get_strategy().num_replicas_in_sync == 1:
                return self.gradients

            return (
                gradient.device_map.select_for_current_replica(gradient.values, replica_context)
                for gradient in self.gradients
                if gradient is not None
            )
        else:
            return self.gradients
