# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import modeling_wo_FFN
import modeling_new_layernorm
import optimization
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

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

flags.DEFINE_integer("save_checkpoints_steps", 10000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("add_GeLU_att", False, "Add GeLU to the FF in attention block")

flags.DEFINE_string("add_weight", None, "Add weights on skip-connection, 'ffn', 'attention', 'all'")

flags.DEFINE_string("weight_type", None, "learn or static")

flags.DEFINE_string("weight_activation", None, "activation function for weights when weight_type == learn")

flags.DEFINE_bool("linear_attention", False, "cancel softmax in self-attention or not")

flags.DEFINE_string("layers_cancel_skip_connection", None, "layers need to cancel skip-connection")

flags.DEFINE_string("layers_use_relu", None, "layers need to use_ReLU")

flags.DEFINE_string("model_type", None, "BERT model type, origin, wo-ffn, new-layernorm")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, add_GeLU_att, add_weight, weight_type, weight_act,
                     linear_attention, cancel_skip_connection, layer_use_relu):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)  # where assigns mode?

        if FLAGS.model_type == 'no-ffn':
            model = modeling_wo_FFN.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
                add_weight=add_weight,
                weight_type=weight_type,
                weight_act=weight_act,
                linear_attention=linear_attention)
        elif FLAGS.model_type == 'new-layernorm':
            model = modeling_new_layernorm.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
                add_weight=add_weight,
                weight_type=weight_type,
                weight_act=weight_act,
                linear_attention=linear_attention)
        elif FLAGS.model_type == 'origin':
            model = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
                add_GeLU_att=add_GeLU_att,
                add_weight=add_weight,
                weight_type=weight_type,
                weight_act=weight_act,
                linear_attention=linear_attention,
                cancel_skip_connection=cancel_skip_connection,
                layer_use_relu=layer_use_relu)
        else:
            raise ValueError("Please specify the model type. ")

        # weights on skip-connection
        all_layer_attention_weight = model.get_all_layer_attention_weight()
        all_layer_ffn_weight = model.get_all_layer_ffn_weight()
        all_layer_attention_weight = [tf.reduce_mean(w) for w in all_layer_attention_weight]
        all_layer_ffn_weight = [tf.reduce_mean(w) for w in all_layer_ffn_weight]

        (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
            bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss + next_sentence_loss

        def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                      masked_lm_weights, next_sentence_example_loss,
                      next_sentence_log_probs, next_sentence_labels):
            """Computes the loss and accuracy of the model."""
            masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                             [-1, masked_lm_log_probs.shape[-1]])
            masked_lm_predictions = tf.argmax(
                masked_lm_log_probs, axis=-1, output_type=tf.int32)
            masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
            masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
            masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
            masked_lm_accuracy = tf.metrics.accuracy(
                labels=masked_lm_ids,
                predictions=masked_lm_predictions,
                weights=masked_lm_weights)
            masked_lm_mean_loss = tf.metrics.mean(
                values=masked_lm_example_loss, weights=masked_lm_weights)

            next_sentence_log_probs = tf.reshape(
                next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
            next_sentence_predictions = tf.argmax(
                next_sentence_log_probs, axis=-1, output_type=tf.int32)
            next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
            next_sentence_accuracy = tf.metrics.accuracy(
                labels=next_sentence_labels, predictions=next_sentence_predictions)
            next_sentence_mean_loss = tf.metrics.mean(
                values=next_sentence_example_loss)

            return {
                "masked_lm_accuracy": masked_lm_accuracy,
                "masked_lm_loss": masked_lm_mean_loss,
                "next_sentence_accuracy": next_sentence_accuracy,
                "next_sentence_loss": next_sentence_mean_loss,
            }

        # load variables from checkpoint if any
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            metric = metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                      masked_lm_weights, next_sentence_example_loss,
                      next_sentence_log_probs, next_sentence_labels)
            metric = {key: value[-1] for key, value in metric.items()}
            metric["total_loss"] = total_loss
            logging_hook = tf.train.LoggingTensorHook(tensors=metric, every_n_iter=100)

            attention_weight_summary = [tf.summary.scalar("attention_weight_layer_{}".format(i), all_layer_attention_weight[i])
                                        for i in range(len(all_layer_attention_weight))]
            ffn_weight_summary = [tf.summary.scalar("ffn_weight_layer_{}".format(i), all_layer_ffn_weight[i])
                                  for i in range(len(all_layer_ffn_weight))]

            # class WeightSessionRunHook(tf.compat.v1.train.SessionRunHook):
            #     def __init__(self, summary, ph, value):
            #         self.summary = summary
            #         self.ph = ph
            #         self.value = value
            #
            #     def begin(self):
            #         self.step = tf.compat.v1.train.get_global_step()
            #
            #     def after_create_session(self, session, coord):
            #         step = session.run(self.step)
            #         value = [session.run(val) for val in self.value]
            #         print(value)
            #         print(type(value[0]))
            #         if step % 100 == 0:
            #             for i in range(len(value)):
            #                 writer = tf.compat.v1.summary.FileWriter(FLAGS.output_dir, session.graph)
            #                 summary_str = session.run(self.summary[i], feed_dict={self.ph: value[i]})
            #                 writer.add_summary(summary_str, step)
            #
            # attention_weight_hook = WeightSessionRunHook(attention_weight_summary,
            #                                              attention_weight_ph,
            #                                              all_layer_attention_weight)
            # ffn_weight_hook = WeightSessionRunHook(ffn_weight_summary,
            #                                        ffn_weight_ph,
            #                                        all_layer_ffn_weight)

            masked_lm_loss_summary = tf.summary.scalar("masked_lm_loss", masked_lm_loss)
            next_sentence_loss_summary = tf.summary.scalar("next_sentence_loss", next_sentence_loss)
            masked_lm_accuracy_summary = tf.summary.scalar("masked_lm_accuracy", metric["masked_lm_accuracy"])
            next_sentence_accuracy_summary = tf.summary.scalar("next_sentence_accuracy", metric["next_sentence_accuracy"])

            summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                     output_dir=FLAGS.output_dir,
                                                     summary_op=tf.summary.merge([masked_lm_loss_summary,
                                                                                  next_sentence_loss_summary,
                                                                                  masked_lm_accuracy_summary,
                                                                                  next_sentence_accuracy_summary] +
                                                                                 attention_weight_summary +
                                                                                 ffn_weight_summary))

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[logging_hook, summary_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels
            ])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


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


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]  # where assigns params?

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

    return input_fn


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


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.io.gfile.makedirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    tf.logging.info("There are {} input TFRecord files. ".format(len(input_files)))

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=10,
        session_config=config,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    layers_cancel_skip_connection = []
    if FLAGS.layers_cancel_skip_connection is not None:
        layers_cancel_skip_connection = list(map(int, FLAGS.layers_cancel_skip_connection.split(',')))
        layers_cancel_skip_connection.sort()
        print("Layers need to cancel skip-connection: ", layers_cancel_skip_connection)

    layers_use_relu = []
    if FLAGS.layers_use_relu is not None:
        layers_use_relu = list(map(int, FLAGS.layers_use_relu.split(',')))
        layers_use_relu.sort()
        print("Layers need to use ReLU: ", layers_use_relu)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        add_GeLU_att=FLAGS.add_GeLU_att,
        add_weight=FLAGS.add_weight,
        weight_type=FLAGS.weight_type,
        weight_act=FLAGS.weight_activation,
        linear_attention=FLAGS.linear_attention,
        cancel_skip_connection=layers_cancel_skip_connection,
        layer_use_relu=layers_use_relu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_type")
    tf.app.run()
