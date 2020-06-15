"""Everytime for one layer, leave the most important head and fine-tune"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('..')
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np
import random

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
                        AxProcessor,
                        MnliMDevAsTestProcessor)
from data_utils import metric_fn, generate_ph_input


# set global seeds
random.seed(12345)
np.random.seed(12345)
tf.compat.v1.random.set_random_seed(12345)

# set the flags
flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
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

flags.DEFINE_integer("hidden_size", 768, "hidden size, i.e. embedding dimension")

flags.DEFINE_bool("do_train", None, "training mode")

flags.DEFINE_bool("load_from_finetuned", None, "training mode")

flags.DEFINE_integer("cur_layer", None, "Current layer needs to mask out heads.")

flags.DEFINE_integer("most_important_head", None, "The most important head in the cur_layer.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 1e-3, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

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
        "ax": AxProcessor,
        "mnlimdevastest": MnliMDevAsTestProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train:
        raise ValueError("At least 'do_train' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.makedirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    print("Current task", task_name)

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    # special handling for mnlimdevastest
    if task_name == 'mnlimdevastest':
        task_name = 'mnlim'

    label_list = processor.get_labels()
    print("Label list of current task", label_list)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_train_examples = len(train_examples)
    num_actual_eval_examples = len(eval_examples)
    print("num_actual_train_examples", num_actual_train_examples)
    print("num_actual_eval_examples", num_actual_eval_examples)

    batch_size = FLAGS.train_batch_size
    epochs = FLAGS.num_train_epochs
    embed_dim = FLAGS.hidden_size  # hidden size, 768 for BERT-base, 512 for BERT-small
    seq_length = FLAGS.max_seq_length
    num_labels = len(label_list)

    # Define some placeholders for the input
    input_ids_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='input_ids')
    input_mask_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='input_mask')
    segment_ids_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='segment_ids')
    label_ids_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name='label_ids')

    tf.compat.v1.logging.info("Running leave the most important head per layer then fine-tune!")

    num_train_steps = num_actual_train_examples // batch_size * epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    cur_layer = FLAGS.cur_layer
    most_important_head = FLAGS.most_important_head
    print("Current most important head:", cur_layer, most_important_head)

    # this placeholder is to control the flag for the dropout
    keep_prob_ph = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, name='is_training')

    # two placeholders for the head coordinates, layer, head
    head_mask_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name='head_mask')
    layer_mask_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name='layer_mask')

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training_ph,
        input_ids=input_ids_ph,  # input_ids,
        input_mask=input_mask_ph,  # input_mask,
        token_type_ids=segment_ids_ph,  # segment_ids,
        use_one_hot_embeddings=False,
        use_estimator=False,
        head_mask=head_mask_ph,
        layer_mask=layer_mask_ph)

    output_layer = model.get_pooled_output()
    output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob_ph)
    output_weights = tf.get_variable(
        "output_weights", [num_labels, embed_dim],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    with tf.compat.v1.variable_scope("loss"):
        # for stsb
        if num_labels == 1:
            logits = tf.squeeze(logits, [-1])
            per_example_loss = tf.square(logits - label_ids_ph)
            loss = tf.reduce_mean(per_example_loss)
        else:
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(label_ids_ph, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

    # metric and summary
    # metric is tf.metric object, (val, op)
    metric = metric_fn(per_example_loss, label_ids_ph, logits, num_labels, task_name)
    metric_name = list(metric.keys())
    metric_val = [m[0] for m in metric.values()]
    metric_op = [m[1] for m in metric.values()]

    log_dir = FLAGS.output_dir + 'layer_{}_head_{}/'.format(cur_layer, most_important_head)

    metric_phs = [tf.compat.v1.placeholder(tf.float32, name="{}_ph".format(key)) for key in metric.keys()]
    summaries = [tf.compat.v1.summary.scalar(key, metric_phs[i]) for i, key in enumerate(metric.keys())]
    train_summary_total = tf.summary.merge(summaries)
    eval_summary_total = tf.summary.merge(summaries)

    init_checkpoint = FLAGS.init_checkpoint
    tvars = tf.compat.v1.trainable_variables()
    var_init = [v for v in tvars if 'output_weights' not in v.name and 'output_bias' not in v.name]
    var_output = [v for v in tvars if 'output_weights' in v.name or "output_bias" in v.name]

    if not FLAGS.load_from_finetuned:
        # Init from Model0
        saver_init = tf.train.Saver(var_init)
    else:
        # Init from Model1
        saver_init = tf.train.Saver(var_init + var_output)

    var_train = var_init + var_output
    print("Training parameters")
    for v in var_train:
        print(v)

    train_op = optimization.create_optimizer(loss=loss,
                                             init_lr=FLAGS.learning_rate,
                                             num_train_steps=num_train_steps,
                                             num_warmup_steps=num_warmup_steps,
                                             use_tpu=False,
                                             tvars=var_train)

    saver_all = tf.train.Saver(var_list=var_init + var_output, max_to_keep=1)

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
        saver_init.restore(sess, init_checkpoint)

        writer = tf.compat.v1.summary.FileWriter(log_dir + 'log/train/', sess.graph)
        writer_eval = tf.compat.v1.summary.FileWriter(log_dir + 'log/eval/')

        # heads need to be masked out in cur_layer
        head_mask = [head for head in range(12) if head != most_important_head]
        layer_mask = [cur_layer for _ in range(11)]

        # if number of eval examples < 1000, just load it directly, or load by batch.
        if num_actual_eval_examples <= 1000:
            eval_input_ids, eval_input_mask, eval_segment_ids, \
            eval_label_ids, eval_is_real_example = generate_ph_input(batch_size=num_actual_eval_examples,
                                                                     seq_length=seq_length,
                                                                     examples=eval_examples,
                                                                     label_list=label_list,
                                                                     tokenizer=tokenizer)

        start_metric = {"eval_{}".format(key): 0 for key in metric_name}
        end_metric = {"eval_{}".format(key): 0 for key in metric_name}

        if FLAGS.do_train:
            tf.logging.info("***** Run training *****")
            step = 1
            for n in range(epochs):

                np.random.shuffle(train_examples)
                num_batch = num_actual_train_examples // batch_size if num_actual_train_examples % batch_size == 0 \
                    else num_actual_train_examples // batch_size + 1
                id = 0

                for b in range(num_batch):

                    input_ids, input_mask, \
                    segment_ids, label_ids, is_real_example = generate_ph_input(batch_size=batch_size,
                                                                                seq_length=seq_length,
                                                                                examples=train_examples,
                                                                                label_list=label_list,
                                                                                tokenizer=tokenizer,
                                                                                train_idx_offset=id)
                    id += batch_size
                    sess.run(metric_vars_initializer)
                    sess.run([train_op] + metric_op, feed_dict={input_ids_ph: input_ids,
                                                                input_mask_ph: input_mask,
                                                                segment_ids_ph: segment_ids,
                                                                label_ids_ph: label_ids,
                                                                is_training_ph: True,
                                                                keep_prob_ph: 0.9,
                                                                head_mask_ph: head_mask,
                                                                layer_mask_ph: layer_mask})
                    train_metric_val = sess.run(metric_val)
                    train_summary_str = sess.run(train_summary_total,
                                                 feed_dict={ph: value for ph, value in
                                                            zip(metric_phs, train_metric_val)})
                    writer.add_summary(train_summary_str, step)

                    if step % 100 == 0 or step % num_batch == 0 or step == 1:
                        # evaluate on dev set

                        if num_actual_eval_examples <= 1000:
                            sess.run(metric_vars_initializer)
                            sess.run(metric_op, feed_dict={input_ids_ph: eval_input_ids,
                                                           input_mask_ph: eval_input_mask,
                                                           segment_ids_ph: eval_segment_ids,
                                                           label_ids_ph: eval_label_ids,
                                                           is_training_ph: False,
                                                           keep_prob_ph: 1,
                                                           head_mask_ph: head_mask,
                                                           layer_mask_ph: layer_mask})
                            eval_metric_val = sess.run(metric_val)
                            eval_summary_str = sess.run(eval_summary_total,
                                                        feed_dict={ph: value for ph, value in
                                                                   zip(metric_phs, eval_metric_val)})
                        else:
                            num_batch_eval = num_actual_eval_examples // batch_size \
                                if num_actual_eval_examples % batch_size == 0 \
                                else num_actual_eval_examples // batch_size + 1
                            id_eval = 0

                            sess.run(metric_vars_initializer)
                            for _ in range(num_batch_eval):
                                eval_input_ids, eval_input_mask, eval_segment_ids, \
                                eval_label_ids, eval_is_real_example = generate_ph_input(batch_size=batch_size,
                                                                                         seq_length=seq_length,
                                                                                         examples=eval_examples,
                                                                                         label_list=label_list,
                                                                                         tokenizer=tokenizer,
                                                                                         train_idx_offset=id_eval)
                                id_eval += batch_size
                                sess.run(metric_op, feed_dict={input_ids_ph: eval_input_ids,
                                                               input_mask_ph: eval_input_mask,
                                                               segment_ids_ph: eval_segment_ids,
                                                               label_ids_ph: eval_label_ids,
                                                               is_training_ph: False,
                                                               keep_prob_ph: 1,
                                                               head_mask_ph: head_mask,
                                                               layer_mask_ph: layer_mask})

                            eval_metric_val = sess.run(metric_val)
                            eval_summary_str = sess.run(eval_summary_total,
                                                        feed_dict={ph: value for ph, value in
                                                                   zip(metric_phs, eval_metric_val)})

                        writer_eval.add_summary(eval_summary_str, step)

                        if step == 1:
                            for key, val in zip(metric_name, eval_metric_val):
                                start_metric["eval_{}".format(key)] = val
                        if step == epochs * num_batch:
                            for key, val in zip(metric_name, eval_metric_val):
                                end_metric["eval_{}".format(key)] = val

                    if step % 100 == 0 or step % num_batch == 0 or step == 1:
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

                        print("Layer {}, leave only one head {} | Epoch: %4d/%4d | Batch: %4d/%4d | {} | {}"
                              .format(cur_layer, most_important_head, train_str, eval_str) % (n, epochs, b, num_batch))

                    if step % num_batch == 0:
                        saver_all.save(sess,
                                       log_dir + 'layer{}_head_{}'.format(cur_layer, most_important_head),
                                       global_step=step)

                    step += 1

            writer.close()
            writer_eval.close()

        print("Start metric", start_metric)
        print("End metric", end_metric)

        with tf.io.gfile.GFile(FLAGS.output_dir + 'results.txt', 'a') as writer:
            eval_start, eval_end = [], []
            for metric in metric_name:
                if metric != 'loss':
                    eval_start.append("{}: %.4f".format(metric) % start_metric["eval_{}".format(metric)])
                    eval_end.append("{}: %.4f".format(metric) % end_metric["eval_{}".format(metric)])

            writer.write("Layer {}, leave only one head {}: Start: {} | End: {}\n"
                         .format(cur_layer,
                                 most_important_head,
                                 ','.join(eval_start),
                                 ','.join(eval_end)))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
