from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import numpy as np
import six
from os.path import join
from six.moves import zip

from absl import flags

import tensorflow as tf
import model_util

def configure_tpu(FLAGS):
    if FLAGS.use_tpu:
        tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        master = tpu_cluster.get_master()
    else:
        tpu_cluster = None
        master = FLAGS.master

    session_config = tf.ConfigProto(allow_soft_placement=True)
    # Uncomment the following line if you hope to monitor GPU RAM growth
    # session_config.gpu_options.allow_growth = True

    if FLAGS.use_tpu:
        strategy = None
        tf.logging.info('Use TPU without distribute strategy.')
    elif FLAGS.num_core_per_host == 1:
        strategy = None
        tf.logging.info('Single device mode.')
    else:
        strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=FLAGS.num_core_per_host)
        tf.logging.info('Use MirroredStrategy with %d devices.',
                        strategy.num_replicas_in_sync)

    per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        model_dir=FLAGS.model_dir,
        session_config=session_config,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations,
            num_shards=FLAGS.num_hosts * FLAGS.num_core_per_host,
            per_host_input_for_training=per_host_input),
        keep_checkpoint_max=FLAGS.max_save,
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.save_steps,
        train_distribute=strategy
    )
    return run_config


def init_from_checkpoint(FLAGS, global_vars=False):
    """initial model from  pre-train model """
    tvars = tf.global_variables() if global_vars else tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if FLAGS.init_checkpoint is not None:
        if FLAGS.model_dir is not None and os.path.exists(FLAGS.model_dir):
            init_checkpoint = FLAGS.model_dir
        else:
            init_checkpoint = FLAGS.init_checkpoint
        tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

        (assignment_map, initialized_variable_names
         ) = model_util.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if FLAGS.use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Log customized initialization
        tf.logging.info("**** Global Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
    return scaffold_fn


def get_assignment_map_from_checkpoint(tvars, init_checkpoint,  num_of_group=1):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)
    init_vars_name = [name for (name, _) in init_vars]

    if num_of_group > 0:
        assignment_map = []
        for gid in range(num_of_group):
            assignment_map.append(collections.OrderedDict())
    else:
        assignment_map = collections.OrderedDict()

    for name in name_to_variable:
        if name in init_vars_name:
            tvar_name = name
        elif (re.sub(r"/group_\d+/", "/group_0/",
                     six.ensure_str(name)) in init_vars_name and
              num_of_group > 1):
            tvar_name = re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name))
        elif (re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
        elif (re.sub(r"/attention_\d+/", "/attention_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/attention_\d+/", "/attention_1/",
                               six.ensure_str(name))
        else:
            tf.logging.info("name %s does not get matched", name)
            continue
        tf.logging.info("name %s match to %s", name, tvar_name)
        if num_of_group > 0:
            group_matched = False
            for gid in range(1, num_of_group):
                if (("/group_" + str(gid) + "/" in name) or
                        ("/ffn_" + str(gid) + "/" in name) or
                        ("/attention_" + str(gid) + "/" in name)):
                    group_matched = True
                    tf.logging.info("%s belongs to %dth", name, gid)
                    assignment_map[gid][tvar_name] = name
            if not group_matched:
                assignment_map[0][tvar_name] = name
        else:
            assignment_map[tvar_name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[six.ensure_str(name) + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def get_train_op(FLAGS, total_loss, update_var_list=None, grads_and_vars=None):
    """ get train operation"""
    global_step = tf.train.get_or_create_global_step()

    # increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
        warmup_lr = (tf.cast(global_step, tf.float32)
                     / tf.cast(FLAGS.warmup_steps, tf.float32)
                     * FLAGS.learning_rate)
    else:
        warmup_lr = 0.0

    # decay the learning rate
    if FLAGS.decay_method == "poly":
        decay_lr = tf.train.polynomial_decay(
            FLAGS.learning_rate,
            global_step=global_step - FLAGS.warmup_steps,
            decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
            end_learning_rate=FLAGS.learning_rate * FLAGS.min_lr_ratio)
    elif FLAGS.decay_method == "cos":
        decay_lr = tf.train.cosine_decay(
            FLAGS.learning_rate,
            global_step=global_step - FLAGS.warmup_steps,
            decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
            alpha=FLAGS.min_lr_ratio)
    else:
        raise ValueError(FLAGS.decay_method)

    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    if FLAGS.weight_decay == 0:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            epsilon=FLAGS.adam_epsilon)
    elif FLAGS.weight_decay > 0 and FLAGS.num_core_per_host == 1:
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            epsilon=FLAGS.adam_epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            weight_decay_rate=FLAGS.weight_decay)
    else:
        raise ValueError("Do not support `weight_decay > 0` with multi-gpu "
                         "training so far.")

    if FLAGS.use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    if grads_and_vars is None:
        grads_and_vars = optimizer.compute_gradients(total_loss, var_list=update_var_list)
    gradients, variables = zip(*grads_and_vars)
    # tf.logging.info("gradient {}".format(gradients[-10:]))
    all_gradients = []
    for gradient in gradients:
        if gradient is not None and "AddN" in gradient.name:
            all_gradients.append(gradient * FLAGS.crf_learning_rate)
        else:
            all_gradients.append(gradient)
    clipped, gnorm = tf.clip_by_global_norm(all_gradients, FLAGS.clip)

    if getattr(FLAGS, "lr_layer_decay_rate", 1.0) != 1.0:
        n_layer = 0
        for i in range(len(clipped)):
            m = re.search(r"model/transformer/layer_(\d+?)/", variables[i].name)
            if not m: continue
            n_layer = max(n_layer, int(m.group(1)) + 1)

        for i in range(len(clipped)):
            for l in range(n_layer):
                if "model/transformer/layer_{}/".format(l) in variables[i].name:
                    abs_rate = FLAGS.lr_layer_decay_rate ** (n_layer - 1 - l)
                    clipped[i] *= abs_rate
                    # tf.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(
                    #     abs_rate, l, variables[i].name))
                    break

    train_op = optimizer.apply_gradients(
        zip(clipped, variables), global_step=global_step)

    # Manually increment `global_step` for AdamWeightDecayOptimizer
    if isinstance(optimizer, AdamWeightDecayOptimizer):
        new_global_step = global_step + 1
        train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    return train_op, learning_rate, gnorm


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"],
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.include_in_weight_decay = include_in_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])

        return tf.group(*assignments, name=name)


