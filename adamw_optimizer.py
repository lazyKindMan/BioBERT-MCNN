# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Adamw for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


class AdamOptimizer(optimizer.Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 weight_decay_rate=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 exclude_from_weight_decay=None,
                 include_in_weight_decay=None,
                 use_locking=False,
                 name="Adamw"):
        """
        This is a multi Gpu version of adamw.
        :param learning_rate:
        :param weight_decay_rate:
        :param beta1:
        :param beta2:
        :param epsilon:
        :param exclude_from_weight_decay:
        :param include_in_weight_decay:
        :param use_locking:
        :param name:
        """
        super(AdamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay_rate = weight_decay_rate
        self._exclude_from_weight_decay = exclude_from_weight_decay
        self._include_in_weight_decay = include_in_weight_decay

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._weight_decay_rate_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "adam_m", self._name)
            self._zeros_slot(v, "adam_v", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        weight_decay_rate = self._call_if_callable(self._weight_decay_rate)
        epsilon = self._call_if_callable(self._epsilon)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._weight_decay_rate_t = ops.convert_to_tensor(
            weight_decay_rate, name="weight_decay_rate")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        weight_decay_rate = math_ops.cast(
            self._weight_decay_rate_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        m = self.get_slot(var, "adam_m")
        v = self.get_slot(var, "adam_v")
        m_t = (tf.multiply(beta1_t, m) + tf.multiply(1.0 - beta1_t, grad))
        m_t = m.assign(m_t, use_locking=self._use_locking)
        v_t = (tf.multiply(beta2_t, v) + tf.multiply(1.0 - beta2_t, tf.square(grad)))
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1. - beta1_power)
        v_t_hat = v_t / (1. - beta2_power)
        update = m_t_hat / (tf.sqrt(v_t_hat) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate * var
        var_update = var - lr * update
        var_update = var.assign(var_update, use_locking=self._use_locking)

        return tf.group(*[var_update, m_t, v_t])

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        weight_decay_rate = math_ops.cast(
            self._weight_decay_rate_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        m = self.get_slot(var, "adam_m")
        v = self.get_slot(var, "adam_v")
        m_t = (tf.multiply(beta1_t, m) + tf.multiply(1.0 - beta1_t, grad))
        m_t = m.assign(m_t, use_locking=self._use_locking)
        v_t = (tf.multiply(beta2_t, v) + tf.multiply(1.0 - beta2_t, tf.square(grad)))
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1. - beta1_power)
        v_t_hat = v_t / (1. - beta2_power)
        update = m_t_hat / (tf.sqrt(v_t_hat) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate * var
        var_update = var - lr * update
        var_update = var.assign(var_update, use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "adam_m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "adam_v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(
            var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x,
                i,
                v,
                use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(
            *update_ops + [update_beta1, update_beta2], name=name_scope)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self._weight_decay_rate:
            return False
        # for r in self._include_in_weight_decay:
        #     if re.search(r, param_name) is not None:
        #         return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    tf.logging.info('Adam WD excludes {}'.format(param_name))
                    return False
        return True
