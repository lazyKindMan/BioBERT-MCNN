import collections
import os
import re
import adamw_optimizer

import tensorflow as tf


def init_from_checkpoint(FLAGS, global_vars=False):
    tvars = tf.global_variables() if global_vars else tf.trainable_variables()
    initialized_variable_names = {}
    if FLAGS.init_checkpoint is not None:
        if FLAGS.init_checkpoint.endswith("latest"):
            ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
            init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        else:
            init_checkpoint = FLAGS.init_checkpoint

        tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Log customized initialization
        tf.logging.info("**** Global Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)


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


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
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

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        # tf.logging.info('original name: %s', name)
        if name not in name_to_variable:
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def get_train_op(FLAGS, total_loss, grads_and_vars=None):
    global_step = tf.train.get_or_create_global_step()

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

    if not FLAGS.use_tpu:
        optimizer = adamw_optimizer.AdamOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=FLAGS.weight_decay,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    else:
        raise Exception("not support tpu")

    tvars = tf.trainable_variables()
    if FLAGS.num_core_per_host > 1:
        grads = tf.gradients(
            total_loss, tvars, colocate_gradients_with_ops=True)
    else:
        grads = tf.gradients(
            total_loss, tvars, colocate_gradients_with_ops=False)

    # This is how the model was pre-trained.
    grads, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)

    train_op = optimizer.apply_gradients(
        list(zip(grads, tvars)), global_step=global_step)

    return train_op, learning_rate, gnorm
