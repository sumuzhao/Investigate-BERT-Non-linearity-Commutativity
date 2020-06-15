import tensorflow as tf


def linear_approximator(input, approximated_layer, hidden_size=768):
    with tf.compat.v1.variable_scope("bert/encoder/layer_{}/non-linearity".format(approximated_layer)):
        y_pred = tf.layers.dense(
            input,
            hidden_size,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    return y_pred


def nonlinear_approximator(input, approximated_layer, hidden_size=768, num_layer=1,
                           use_dropout=False, dropout_p=0.2):
    with tf.compat.v1.variable_scope("bert/encoder/layer_{}/non-linearity".format(approximated_layer)):
        prev_output = input
        for layer in range(num_layer):
            layer_input = prev_output
            with tf.compat.v1.variable_scope("nla_layer_{}".format(layer)):
                y_pred = tf.layers.dense(
                    layer_input,
                    hidden_size,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if use_dropout:
                    y_pred = tf.nn.dropout(y_pred, rate=dropout_p)
            prev_output = y_pred
    return y_pred

