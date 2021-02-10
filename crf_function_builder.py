from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# import albert_modeling as modeling
import bert_base.modeling as modeling
from lstm_crf_layer import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers

def get_crf_outputs(FLAGS, features, is_training, top_three_convolution_kernel):
    input_ids = features["input_ids"]
    segment_ids = features["segment_ids"]
    input_mask = features["input_mask"]
    label_ids = features["label_ids"]
    is_start_label = features['is_start_label']
    weight_matrixs = features['weight_matrix']
    words_label = features['words_label']
    label_x = features["label_x"]
    is_weight_focus = features["is_weight_focus"]
    # token_weight = features["token_weight"]
    max_seq_length = FLAGS.max_seq_length
    batch_size = tf.shape(input_ids)[0]

    classes = FLAGS.crf_classes

    if FLAGS.label_mode == "BL":
        labels = label_x
        classes += 1
        mask = features["is_start_label"]
    if FLAGS.label_mode == "WPL":
        labels = label_ids
        mask = features["input_mask"]

    model_config = modeling.BertConfig.from_json_file(FLAGS.model_config_path)
    bert_model = modeling.BertModel(config=model_config,
                                    is_training=is_training,
                                    input_ids=input_ids,
                                    input_mask=input_mask,
                                    token_type_ids=segment_ids
                                    )
    # get bert last hidden layer output for crf
    output_layer = bert_model.get_sequence_output()
    # 卷积特征
    hidden_size = output_layer.shape[-1].value
    if not FLAGS.no_cnn:
        with tf.name_scope("mul_cnn"):
            cnn_outputs = []
            filter_size = hidden_size / len(top_three_convolution_kernel)
            for kernel_size in top_three_convolution_kernel:
                # CNN layer
                conv = tf.layers.conv1d(output_layer, filter_size, kernel_size,
                                        name='conv-%s' % kernel_size, padding='same')
                cnn_outputs.append(conv)
            # connect the output
            output_layer = tf.concat(cnn_outputs, -1)
    # full-connected layer
    output_weight = tf.get_variable(
        "output_weights", [classes, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [classes], initializer=tf.zeros_initializer()
    )
    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    start_logits = tf.matmul(output_layer, output_weight, transpose_b=True)
    start_logits = tf.nn.bias_add(start_logits, output_bias)
    start_logits = tf.reshape(start_logits, [-1, FLAGS.max_seq_length, classes])
    with tf.variable_scope("crf_layer"):
        start_logits = tf.matmul(tf.reshape(weight_matrixs, shape=(batch_size, max_seq_length, max_seq_length)), start_logits)
        mask = tf.cast(mask, tf.float32)
        if FLAGS.no_crf:
            log_probs = tf.nn.log_softmax(start_logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=classes, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            total_loss = tf.reduce_sum(tf.reshape(mask * per_example_loss, [-1]))
            probabilities = tf.nn.softmax(start_logits, axis=-1)
            logits = tf.argmax(probabilities, axis=-1)
        train_layer = FLAGS.trainable_layer
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_var_list = []
        if train_layer >= model_config.num_hidden_layers:
            update_var_list = trainable_vars
        else:
            notrainable_layers = []
            for i in range(model_config.num_hidden_layers - train_layer):
                notrainable_layers.append("bert/encoder/layer_{}".format(i))
            tf.logging.info("the notrainable layer is {}".format(str(notrainable_layers)))
            notrainable_set = set()
            for tvar in trainable_vars:
                f = True
                for notrainable_layer in notrainable_layers:
                    if notrainable_layer == "/".join(tvar.name.split("/")[0:3]):
                        f = False
                        notrainable_set.add(notrainable_layer)
                        break
                if f:
                    update_var_list.append(tvar)
            tf.logging.info("the notrainable layer num is {}".format(str(len(notrainable_set))))
        if FLAGS.no_crf:
            return total_loss, per_example_loss, logits, labels, input_mask, is_start_label, update_var_list
        else:
            return total_loss, per_example_loss, pred_ids, labels, input_mask, is_start_label, update_var_list


def get_initializer(FLAGS):
    """Get variable initializer for the Fully connected layer which between last bert hidden layer and  crf"""
    if FLAGS.init == "uniform":
        initializer = tf.initializers.random_uniform(
            minval=-FLAGS.init_range,
            maxval=FLAGS.init_range,
            seed=None)
    elif FLAGS.init == "normal":
        initializer = tf.initializers.random_normal(
            stddev=FLAGS.init_std,
            seed=None)
    else:
        raise ValueError("Initializer {} not supported".format(FLAGS.init))
    return initializer
