"""BERT finetuning runner on GLUE"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
import collections
import os
import modeling
import modeling_wo_FFN
import modeling_new_layernorm
import optimization
import tokenization
import tensorflow as tf

from data_utils import (ColaProcessor,
                        MnliMProcessor,
                        MnliMMProcessor,
                        MrpcProcessor,
                        QnliProcessor,
                        QqpProcessor,
                        RteProcessor,
                        Sst2Processor,
                        StsbProcessor,
                        WnliProcessor,
                        AxProcessor)
from data_utils import PaddingInputExample, convert_single_example, metric_fn, standard_file_name


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("train_step", 1000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer(
    "warmup_step", 0,
    "number of steps to perform linear learning rate warmup for.")

flags.DEFINE_bool("add_GeLU_att", False, "Add GeLU to the FF in attention block")

flags.DEFINE_string("add_weight", None, "Add weights on skip-connection, 'ffn', 'attention', 'all'")

flags.DEFINE_string("weight_type", None, "learn or static")

flags.DEFINE_string("weight_activation", None, "activation function for weights when weight_type == learn")

flags.DEFINE_bool("linear_attention", False, "cancel softmax in self-attention or not")

flags.DEFINE_string("layers_cancel_skip_connection", None, "layers need to cancel skip-connection")

flags.DEFINE_string("layers_use_relu", None, "layers need to use_ReLU")

flags.DEFINE_string("model_type", None, "BERT model type, origin, wo-ffn, new-layernorm")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

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


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature, _ = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        # sts-b is regression problem
        if len(label_list) > 1:
            features["label_ids"] = create_int_feature([feature.label_id])
        else:
            features["label_ids"] = create_float_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, task):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64) if task != 'stsb' else tf.FixedLenFeature([], tf.float32),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

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

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, add_GeLU_att,
                 add_weight, weight_type, weight_act, linear_attention, cancel_skip_connection,
                 layer_use_relu):
    """Creates a classification model."""
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

    # In the demo, we are doing a simple classification task on the entire segment.
    # If you want to use the token-level output, use model.get_sequence_output() instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        # for sts-b
        if num_labels == 1:
            logits = tf.squeeze(logits, [-1])
            per_example_loss = tf.square(logits - labels)
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, logits, None

        else:
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task, add_GeLU_att,
                     add_weight, weight_type, weight_act, linear_attention, cancel_skip_connection,
                     layer_use_relu):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, add_GeLU_att, add_weight, weight_type, weight_act,
            linear_attention, cancel_skip_connection, layer_use_relu)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            assignment_map, initialized_variable_names = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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

            # metrics = metric_fn(per_example_loss, label_ids, logits, num_labels, task)
            # train_metrics = {'Batch_{}'.format(name.split('_')[-1]): metrics[name][-1] for name in metrics.keys()}
            # train_metrics['Total_loss'] = total_loss
            # logging_hook = tf.train.LoggingTensorHook(tensors=train_metrics, every_n_iter=100)
            #
            # metric_sum = []
            # for key in train_metrics.keys():
            #     metric_sum.append(tf.summary.scalar(key, train_metrics[key]))
            # summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=FLAGS.output_dir,
            #                                          summary_op=tf.summary.merge(metric_sum))

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
                # training_hooks=[logging_hook, summary_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, num_labels, task, True])
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output = {}
            if num_labels > 1:
                predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
                output["probabilities"] = probabilities
            else:
                predictions = logits
            output["predictions"] = predictions

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=output,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cola": ColaProcessor,
        "mnlim": MnliMProcessor,
        "mnlimm": MnliMMProcessor,
        "mrpc": MrpcProcessor,
        "qnli": QnliProcessor,
        "qqp": QqpProcessor,
        "rte": RteProcessor,
        "sst2": Sst2Processor,
        "stsb": StsbProcessor,
        "wnli": WnliProcessor,
        "ax": AxProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # for submission folder path
    submit_dir = os.path.join(FLAGS.output_dir, "submission")
    if not tf.io.gfile.exists(submit_dir):
        tf.io.gfile.makedirs(submit_dir)

    task_name = FLAGS.task_name.lower()

    FLAGS.output_dir = os.path.join(FLAGS.output_dir, "{}_output".format(task_name))
    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

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
        keep_checkpoint_max=1,
        session_config=config,
        log_step_count_steps=100,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    # here directly feed the steps instead of epochs
    num_train_steps = FLAGS.train_step
    num_warmup_steps = FLAGS.warmup_step
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        # num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    layers_cancel_skip_connection = []
    if FLAGS.layers_cancel_skip_connection is not None and FLAGS.layers_cancel_skip_connection != '':
        layers_cancel_skip_connection = list(map(int, FLAGS.layers_cancel_skip_connection.split(',')))
        layers_cancel_skip_connection.sort()
        print("Layers need to cancel skip-connection: ", layers_cancel_skip_connection)

    layers_use_relu = []
    if FLAGS.layers_use_relu is not None and FLAGS.layers_use_relu != '':
        layers_use_relu = list(map(int, FLAGS.layers_use_relu.split(',')))
        layers_use_relu.sort()
        print("Layers need to use ReLU: ", layers_use_relu)

    # add task parameter, GLUE tasks have different metrics.
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        task=task_name,
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
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not tf.io.gfile.exists(train_file):
            file_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            task=task_name)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        # eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        # eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        # if not tf.io.gfile.exists(eval_file):
        #     file_based_convert_examples_to_features(
        #         eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        # eval_input_fn = file_based_input_fn_builder(
        #     input_file=eval_file,
        #     seq_length=FLAGS.max_seq_length,
        #     is_training=False,
        #     drop_remainder=eval_drop_remainder,
        #     task=task_name)
        #
        # # TODO: not sure if this is correct...
        # class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
        #     def __init__(self, estimator, input_fn):
        #         self.estimator = estimator
        #         self.input_fn = input_fn
        #
        #     def after_save(self, session, global_step):
        #         self.estimator.evaluate(input_fn=self.input_fn)
        #
        # saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.output_dir, save_steps=10,
        #                                           listeners=[EvalCheckpointSaverListener(estimator, eval_input_fn)])
        # estimator.train(
        #     input_fn=train_input_fn,
        #     max_steps=num_train_steps,
        #     hooks=[saver_hook])

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if not tf.io.gfile.exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            task=task_name)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        if not tf.io.gfile.exists(predict_file):
            file_based_convert_examples_to_features(predict_examples, label_list,
                                                    FLAGS.max_seq_length, tokenizer,
                                                    predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            task=task_name)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        submit_predict_file = os.path.join(FLAGS.output_dir, "{}.tsv".format(standard_file_name[task_name]))
        writer_output = tf.io.gfile.GFile(output_predict_file, "w")
        writer_submit = tf.io.gfile.GFile(submit_predict_file, 'w')

        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        writer_submit.write("ID \t Label \n")
        for (i, prediction) in enumerate(result):
            if i >= num_actual_predict_examples:
                break
            if task_name != 'stsb':
                label = label_list[prediction["predictions"]]
                probabilities = prediction["probabilities"]
                writer_output.write("\t".join(str(class_probability) for class_probability in probabilities) + "\n")
                writer_submit.write("{} \t {} \n".format(num_written_lines, label))
            else:
                label = prediction["predictions"]
                writer_output.write("{} \t {} \n".format(num_written_lines, label))
                writer_submit.write("{} \t {} \n".format(num_written_lines, label))
            num_written_lines += 1
        writer_output.close()
        writer_submit.close()
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_type")
    tf.app.run()
