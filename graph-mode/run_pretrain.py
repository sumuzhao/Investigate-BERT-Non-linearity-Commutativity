"""Pre-train BERT Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import tensorflow as tf
import numpy as np
import random
import os

import sys
sys.path.append('..')
import optimization
import modeling
import modeling_wo_FFN
import modeling_new_layernorm


# set global seeds
random.seed(12345)
np.random.seed(12345)
tf.compat.v1.random.set_random_seed(12345)

# set the flags
flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files for training (can be a glob or comma separated).")

flags.DEFINE_string(
    "eval_input_file", None,
    "Input TF example files for evaluation (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 100000, "How often to save the model checkpoint.")

flags.DEFINE_integer("print_freq", 1000, "How many steps to print results. ")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_string("add_weight", None, "Add weights on skip-connection, 'ffn', 'attention', 'all'")

flags.DEFINE_string("weight_type", None, "learn or static")

flags.DEFINE_string("weight_activation", None, "activation function for weights when weight_type == learn")

flags.DEFINE_bool("linear_attention", False, "cancel softmax in self-attention or not")

flags.DEFINE_string("layers_cancel_skip_connection", None, "layers need to cancel skip-connection")

flags.DEFINE_string("model_type", None, "BERT model type, origin, wo-ffn, new-layernorm")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_reader(input_files, max_seq_length, max_predictions_per_seq,
                 is_training, batch_size, num_cpu_threads=4):
    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=100)
    else:
        d = tf.data.TFRecordDataset(input_files)
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels):
    """Computes the loss and accuracy of the model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(labels=masked_lm_ids,
                                             predictions=masked_lm_predictions,
                                             weights=masked_lm_weights,
                                             name='masked_lm_accuracy')
    masked_lm_mean_loss = tf.metrics.mean(values=masked_lm_example_loss,
                                          weights=masked_lm_weights,
                                          name='masked_lm_loss')

    next_sentence_log_probs = tf.reshape(next_sentence_log_probs,
                                         [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis=-1, output_type=tf.int32)
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.metrics.accuracy(labels=next_sentence_labels,
                                                 predictions=next_sentence_predictions,
                                                 name='next_sentence_accuracy')
    next_sentence_mean_loss = tf.metrics.mean(values=next_sentence_example_loss,
                                              name='next_sentence_loss')

    return {"masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss}


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if not FLAGS.do_train and \
       not FLAGS.do_eval:
        raise ValueError("At least one of 'do_train' or 'do_eval' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.io.gfile.makedirs(FLAGS.output_dir)

    # load TFRecord files
    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
        train_input_files.extend(tf.gfile.Glob(input_pattern))
    eval_input_files = []
    for input_pattern in FLAGS.eval_input_file.split(","):
        eval_input_files.extend(tf.gfile.Glob(input_pattern))

    seq_length = FLAGS.max_seq_length

    # create dataset
    train_dataset = input_reader(input_files=train_input_files,
                                 max_seq_length=seq_length,
                                 max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                 is_training=True,
                                 batch_size=FLAGS.train_batch_size)
    eval_dataset = input_reader(input_files=eval_input_files,
                                max_seq_length=seq_length,
                                max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                is_training=False,
                                batch_size=FLAGS.eval_batch_size)

    # handle constructions. Handle allows us to feed data from different dataset by providing a parameter in feed_dict
    handle_ph = tf.placeholder(tf.string, shape=[], name="handle")
    output_shapes = {key: tf.TensorShape([None, None]) for key in train_dataset.output_shapes}
    iterator = tf.data.Iterator.from_string_handle(string_handle=handle_ph,
                                                   output_types=train_dataset.output_types,
                                                   output_shapes=output_shapes)
    next_element = iterator.get_next()

    # create iterator
    train_iterator = train_dataset.make_initializable_iterator()
    eval_iterator = eval_dataset.make_initializable_iterator()

    # take the features out of the next element
    input_ids = next_element["input_ids"]
    input_mask = next_element["input_mask"]
    segment_ids = next_element["segment_ids"]
    masked_lm_positions = next_element["masked_lm_positions"]
    masked_lm_ids = next_element["masked_lm_ids"]
    masked_lm_weights = next_element["masked_lm_weights"]
    next_sentence_labels = next_element["next_sentence_labels"]

    tf.compat.v1.logging.info("Running pre-training!")

    layers_cancel_skip_connection = []
    if FLAGS.layers_cancel_skip_connection is not None:
        layers_cancel_skip_connection = list(map(int, FLAGS.layers_cancel_skip_connection.split(',')))
        layers_cancel_skip_connection.sort()
        print("Layers need to cancel skip-connection: ", layers_cancel_skip_connection)

    # this placeholder is to control the flag for the dropout
    is_training_ph = tf.compat.v1.placeholder(tf.bool, name='is_training')

    if FLAGS.model_type == 'no-ffn':
        model = modeling_wo_FFN.BertModel(
            config=bert_config,
            is_training=is_training_ph,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=FLAGS.use_tpu,
            add_weight=FLAGS.add_weight,
            weight_type=FLAGS.weight_type,
            weight_act=FLAGS.weight_activation,
            linear_attention=FLAGS.linear_attention,
            use_estimator=False)
    elif FLAGS.model_type == 'new-layernorm':
        model = modeling_new_layernorm.BertModel(
            config=bert_config,
            is_training=is_training_ph,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=FLAGS.use_tpu,
            add_weight=FLAGS.add_weight,
            weight_type=FLAGS.weight_type,
            weight_act=FLAGS.weight_activation,
            linear_attention=FLAGS.linear_attention,
            use_estimator=False)
    elif FLAGS.model_type == 'origin':
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training_ph,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=FLAGS.use_tpu,
            add_weight=FLAGS.add_weight,
            weight_type=FLAGS.weight_type,
            weight_act=FLAGS.weight_activation,
            linear_attention=FLAGS.linear_attention,
            cancel_skip_connection=layers_cancel_skip_connection,
            use_estimator=False)
    else:
        raise ValueError("Please specify the model type. ")

    masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs = get_masked_lm_output(
        bert_config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids, masked_lm_weights)

    next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs = get_next_sentence_output(
        bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    # metric and summary
    # metric is tf.metric object, (val, op)
    metric = metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights,
                       next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels)
    metric_name = list(metric.keys())
    metric_val = [m[0] for m in metric.values()]
    metric_op = [m[1] for m in metric.values()]

    metric_phs = [tf.compat.v1.placeholder(tf.float32, name="{}_ph".format(key)) for key in metric.keys()]
    summaries = [tf.compat.v1.summary.scalar(key, metric_phs[i]) for i, key in enumerate(metric.keys())]
    train_summary_total = tf.summary.merge(summaries)
    eval_summary_total = tf.summary.merge(summaries)

    log_dir = FLAGS.output_dir

    # initialze weights
    tvars = tf.compat.v1.trainable_variables()
    saver_init = tf.train.Saver(tvars)
    print("Training parameters")
    for v in tvars:
        print(v)

    train_op = optimization.create_optimizer(loss=total_loss,
                                             init_lr=FLAGS.learning_rate,
                                             num_train_steps=FLAGS.num_train_steps,
                                             num_warmup_steps=FLAGS.num_warmup_steps,
                                             use_tpu=False,
                                             tvars=tvars)

    saver_all = tf.train.Saver(var_list=tvars, max_to_keep=50)

    # Isolate the variables stored behind the scenes by the metric operation
    var_metric = []
    for key in metric.keys():
        var_metric.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=key))
    # Define initializer to initialize/reset running variables
    metric_vars_initializer = tf.variables_initializer(var_list=var_metric)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.init_checkpoint:
            saver_init.restore(sess, FLAGS.init_checkpoint)

        start_metric = {"eval_{}".format(key): 0 for key in metric_name}
        if FLAGS.do_train:
            tf.logging.info("***** Run training *****")

            writer = tf.compat.v1.summary.FileWriter(log_dir + 'log/train/', sess.graph)
            writer_eval = tf.compat.v1.summary.FileWriter(log_dir + 'log/eval/')

            # generate handle for train/eval
            train_handle = sess.run(train_iterator.string_handle())
            eval_handle = sess.run(eval_iterator.string_handle())

            # initialize the iterator of training dataset
            sess.run(train_iterator.initializer)

            step = 1
            for _ in range(FLAGS.num_train_steps):
                sess.run(metric_vars_initializer)
                sess.run([train_op] + metric_op, feed_dict={handle_ph: train_handle,
                                                            is_training_ph: True})
                train_metric_val = sess.run(metric_val)
                train_summary_str = sess.run(train_summary_total,
                                             feed_dict={ph: value for ph, value in
                                                        zip(metric_phs, train_metric_val)})
                writer.add_summary(train_summary_str, step)

                if step % FLAGS.print_freq == 0 or step == 1:
                    # initialize the iterator of eval dataset
                    sess.run(eval_iterator.initializer)

                    sess.run(metric_vars_initializer)
                    try:
                        for _ in range(16):
                            sess.run(metric_op, feed_dict={handle_ph: eval_handle,
                                                           is_training_ph: False})
                    except tf.errors.OutOfRangeError:
                        pass
                    eval_metric_val = sess.run(metric_val)
                    eval_summary_str = sess.run(eval_summary_total,
                                                feed_dict={ph: value for ph, value in
                                                           zip(metric_phs, eval_metric_val)})
                    writer_eval.add_summary(eval_summary_str, step)

                if step % FLAGS.print_freq == 0 or step == 1:
                    train_metric_list = []
                    for i in range(len(train_metric_val)):
                        if metric_name[i] == 'loss':
                            train_metric_list.append("{}: %2.4f".format(metric_name[i]) % train_metric_val[i])
                        else:
                            train_metric_list.append("{}: %.4f".format(metric_name[i]) % train_metric_val[i])
                    train_str = 'Train ' + '|'.join(train_metric_list)

                    eval_metric_list = []
                    for i in range(len(eval_metric_val)):
                        if metric_name[i] == 'loss':
                            eval_metric_list.append("{}: %2.4f".format(metric_name[i]) % eval_metric_val[i])
                        else:
                            eval_metric_list.append("{}: %.4f".format(metric_name[i]) % eval_metric_val[i])
                    eval_str = 'Eval ' + '|'.join(eval_metric_list)

                    print("Pre-training | Step: %4d/%4d | {} | {}"
                          .format(train_str, eval_str) % (step, FLAGS.num_train_steps))

                if step % FLAGS.save_checkpoints_steps == 0 or step == 1:
                    saver_all.save(sess, log_dir + 'bert_model', global_step=step)

                step += 1

            writer.close()
            writer_eval.close()


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_type")
    tf.compat.v1.app.run()
