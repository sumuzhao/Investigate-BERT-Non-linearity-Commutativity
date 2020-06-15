"""Train non-linearity for certain parts of BERT"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('..')
import modeling
import tokenization
import tensorflow as tf
import numpy as np
import random
from approximator import approximator
from scipy.spatial.distance import cosine, euclidean

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
from data_utils import generate_ph_input

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

flags.DEFINE_string("layers", None, "only digits if single layer, splited by ',' if multiple layers, e.g. 1,2,3")

flags.DEFINE_string("approximator_setting", None, "non-linearity settings, HS_MLP, HS*4+HS_MLP, HS*4_FFGeLU")

flags.DEFINE_string("approximate_part", None, "what part to approximate, attention, ffn, encoder")

flags.DEFINE_string("layers_cancel_skip_connection", None, "layers need to cancel skip-connection")

flags.DEFINE_string("layers_use_relu", None, "layers need to use_ReLU")

flags.DEFINE_string("loss", None, "loss type when training linear non-linearity")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 1e-3, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")

flags.DEFINE_bool("use_nonlinear_approximator", False, "Whether to use non-linear approximators. ")

flags.DEFINE_bool("use_dropout", False, "Whether to use dropout in approximators. ")


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

    # Define some placeholders for the input
    input_ids_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='input_ids')
    input_mask_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='input_mask')
    segment_ids_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='segment_ids')

    # train an independent linear model to approximate the projection of MLP
    tf.compat.v1.logging.info("Training Approximator!")

    # get the layer(s) which need replacement
    if FLAGS.layers is None:
        raise ValueError("In training non-linearity experiments, layers must not be None. ")
    layer_folder_name = FLAGS.layers
    layers = list(map(int, FLAGS.layers.split(',')))
    layers.sort()
    if len(layers) != 1:
        raise ValueError("Here it allows only one single layer. ")
    approximated_layer = layers[0]
    print("Current approximated layer: ", approximated_layer)
    approximator_setting = FLAGS.approximator_setting

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

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids_ph,  # input_ids,
        input_mask=input_mask_ph,  # input_mask,
        token_type_ids=segment_ids_ph,  # segment_ids,
        use_one_hot_embeddings=False,
        approximator_setting=approximator_setting,
        cancel_skip_connection=layers_cancel_skip_connection,
        layer_use_relu=layers_use_relu)

    # define the input and output according to the approximated part
    if FLAGS.approximate_part == 'mlp':
        # only FFGeLU and FF in FFN, without dropout and layernorm
        x = model.get_all_layer_tokens_beforeMLP()[approximated_layer]
        y = model.get_all_layer_tokens_afterMLP()[approximated_layer]
    elif FLAGS.approximate_part == 'ffgelu':
        # only FFGeLU in FFN
        x = model.get_all_layer_tokens_beforeMLP()[approximated_layer]
        y = model.get_all_layer_tokens_after_FFGeLU()[approximated_layer]
    elif FLAGS.approximate_part == 'self_attention':
        # only self-attention part, without linear layer, dropout and layernorm
        x = model.get_all_encoder_layers()[approximated_layer - 1] if approximated_layer > 0 else model.get_embedding_output()
        y = model.get_all_head_output()[approximated_layer]
    elif FLAGS.approximate_part == 'self_attention_ff':
        # only self-attention + linear part part, without dropout and layernorm
        x = model.get_all_encoder_layers()[approximated_layer - 1] if approximated_layer > 0 else model.get_embedding_output()
        y = model.get_all_attention_before_dropout()[approximated_layer]
    elif FLAGS.approximate_part == 'ff_after_self_attention':
        # only the linear layer after self-attention
        x = model.get_all_head_output()[approximated_layer]
        y = model.get_all_attention_before_dropout()[approximated_layer]
    elif FLAGS.approximate_part == 'attention_before_ln':
        # only self-attention + linear part part, before layernorm
        x = model.get_all_encoder_layers()[approximated_layer - 1] if approximated_layer > 0 else model.get_embedding_output()
        y = model.get_all_attention_before_layernorm()[approximated_layer]
    elif FLAGS.approximate_part == 'attention':
        # whole attention block including dropout and layernorm
        x = model.get_all_encoder_layers()[approximated_layer - 1] if approximated_layer > 0 else model.get_embedding_output()
        y = model.get_all_layer_tokens_beforeMLP()[approximated_layer]
    elif FLAGS.approximate_part == 'ffn':
        # whole FFN block including dropout and layernorm
        x = model.get_all_layer_tokens_beforeMLP()[approximated_layer]
        y = model.get_all_encoder_layers()[approximated_layer]
    elif FLAGS.approximate_part == 'ffn_before_ln':
        # only FFGeLU and FF in FFN, before layernorm
        x = model.get_all_layer_tokens_beforeMLP()[approximated_layer]
        y = model.get_all_ffn_before_layernorm()[approximated_layer]
    elif FLAGS.approximate_part == 'encoder':
        # whole encoder including dropout and layernorm
        x = model.get_all_encoder_layers()[approximated_layer - 1] if approximated_layer > 0 else model.get_embedding_output()
        y = model.get_all_encoder_layers()[approximated_layer]
    else:
        raise ValueError("Need to specify correct value. ")

    if FLAGS.approximator_setting in ['HS_MLP', 'HS*4+HS_MLP', 'HS_Self_Attention', 'HS_Self_Attention_FF',
                                      'HS_FFN', 'HS_Attention', 'HS_Encoder',
                                      'HS_Attention_Before_LN', 'HS_FFN_Before_LN',
                                      'HS_FF_After_Self_Attention']:
        approximator_dim = embed_dim
    else:  # HS*4_FFGeLU
        approximator_dim = embed_dim * 4

    x = tf.reshape(x, [-1, seq_length, embed_dim])
    y = tf.reshape(y, [-1, seq_length, approximator_dim])

    # non-linearity
    if not FLAGS.use_nonlinear_approximator:
        y_pred = approximator.linear_approximator(input=x, approximated_layer=approximated_layer,
                                                  hidden_size=approximator_dim)
    else:
        y_pred = approximator.nonlinear_approximator(input=x, approximated_layer=approximated_layer,
                                                     hidden_size=approximator_dim, num_layer=1,
                                                     use_dropout=FLAGS.use_dropout, dropout_p=0.2)

    # with tf.compat.v1.variable_scope("bert/encoder/layer_{}/non-linearity".format(approximated_layer)):
    #     y_pred = tf.layers.dense(
    #         x,
    #         approximator_dim,
    #         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    # NOTE, here is alias of cosine_proximity, [0, 1]. Not the TF2.0 version, which is [-1. 0].
    similarity_token = tf.keras.losses.cosine_similarity(y, y_pred, axis=-1)
    l2_distance_token = tf.norm(y - y_pred, ord=2, axis=-1)
    # loss: similarity or mean squared error or l2 norm
    # when using cosine similarity, the target becomes minimizing to 0.
    per_example_loss = 1. - tf.reduce_mean(similarity_token, axis=-1) \
        if FLAGS.loss == 'cosine' \
        else (tf.reduce_mean(tf.keras.losses.mse(y, y_pred), axis=-1)
              if FLAGS.loss == 'mse'
              else (tf.reduce_mean(l2_distance_token, axis=-1)
                    if FLAGS.loss == 'l2'
                    else 0.0))

    loss = tf.reduce_mean(per_example_loss, axis=-1)
    similarity = tf.reduce_mean(tf.reduce_mean(similarity_token, axis=-1), axis=-1)
    l2_distance = tf.reduce_mean(tf.reduce_mean(l2_distance_token, axis=-1), axis=-1)

    # for summary
    loss_ph = tf.compat.v1.placeholder(tf.float32, name='loss')
    similarity_ph = tf.compat.v1.placeholder(tf.float32, name='similarity')
    l2_distance_ph = tf.compat.v1.placeholder(tf.float32, name='l2_distance')
    loss_sum = tf.summary.scalar("loss", loss_ph)
    similarity_sum = tf.summary.scalar("similarity", similarity_ph)
    l2_distance_sum = tf.summary.scalar("l2_distance", l2_distance_ph)
    train_summary_total = tf.summary.merge([loss_sum, similarity_sum, l2_distance_sum])
    eval_summary_total = tf.summary.merge([loss_sum, similarity_sum, l2_distance_sum])

    log_dir = FLAGS.output_dir + 'layer_{}/'.format(layer_folder_name)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # load weights
    init_checkpoint = FLAGS.init_checkpoint
    tvars = tf.compat.v1.trainable_variables()
    var_init = [var for var in tvars if "non-linearity" not in var.name]
    saver_init = tf.train.Saver(var_init)

    # take variables related to approximators
    var_approximator = [var for var in tvars if "non-linearity" in var.name]
    print("Training variables")
    for var in var_approximator:
        print(var)

    # define optimizer op and saver
    optimizer_op = optimizer.minimize(loss=loss, var_list=var_approximator)
    saver_approximator = tf.train.Saver(var_list=var_approximator, max_to_keep=1)

    # add this GPU settings
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver_init.restore(sess, init_checkpoint)

        writer = tf.compat.v1.summary.FileWriter(log_dir + 'log/train/', sess.graph)
        writer_eval = tf.compat.v1.summary.FileWriter(log_dir + 'log/eval/')

        # if number of eval examples < 1000, just load it directly, or load by batch.
        if num_actual_eval_examples <= 1000:
            eval_input_ids, eval_input_mask, eval_segment_ids, \
            eval_label_ids, eval_is_real_example, \
            eval_input_tokens = generate_ph_input(batch_size=num_actual_eval_examples,
                                                  seq_length=seq_length,
                                                  examples=eval_examples,
                                                  label_list=label_list,
                                                  tokenizer=tokenizer,
                                                  return_tokens=True)

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

                train_loss, train_similarity, train_l2_distance,  _ = sess.run(
                    [loss, similarity, l2_distance, optimizer_op],
                      feed_dict={input_ids_ph: input_ids,
                                 input_mask_ph: input_mask,
                                 segment_ids_ph: segment_ids})
                train_summary_str = sess.run(train_summary_total, feed_dict={loss_ph: train_loss,
                                                                             similarity_ph: train_similarity,
                                                                             l2_distance_ph: train_l2_distance})
                writer.add_summary(train_summary_str, step)

                # evaluate on dev set
                if step % 100 == 0 or step % num_batch == 0 or step == 1:

                    if num_actual_eval_examples <= 1000:
                        eval_loss, eval_similarity, eval_l2_distance = sess.run(
                            [loss, similarity, l2_distance],
                            feed_dict={input_ids_ph: eval_input_ids,
                                       input_mask_ph: eval_input_mask,
                                       segment_ids_ph: eval_segment_ids})
                        eval_summary_str = sess.run(eval_summary_total, feed_dict={loss_ph: eval_loss,
                                                                                   similarity_ph: eval_similarity,
                                                                                   l2_distance_ph: eval_l2_distance})

                    else:

                        num_batch_eval = num_actual_eval_examples // batch_size \
                            if num_actual_eval_examples % batch_size == 0 \
                            else num_actual_eval_examples // batch_size + 1
                        id_eval = 0
                        acc_loss, acc_similarity, acc_l2_distance = 0, 0, 0

                        for _ in range(num_batch_eval):
                            eval_input_ids, eval_input_mask, eval_segment_ids, \
                            eval_label_ids, eval_is_real_example = generate_ph_input(batch_size=batch_size,
                                                                                     seq_length=seq_length,
                                                                                     examples=eval_examples,
                                                                                     label_list=label_list,
                                                                                     tokenizer=tokenizer,
                                                                                     train_idx_offset=id_eval)
                            id_eval += batch_size

                            eval_loss, eval_similarity, eval_l2_distance = sess.run(
                                [loss, similarity, l2_distance], feed_dict={input_ids_ph: eval_input_ids,
                                                                            input_mask_ph: eval_input_mask,
                                                                            segment_ids_ph: eval_segment_ids})
                            acc_loss += eval_loss
                            acc_similarity += eval_similarity
                            acc_l2_distance += eval_l2_distance

                        eval_loss = acc_loss / num_batch_eval
                        eval_similarity = acc_similarity / num_batch_eval
                        eval_l2_distance = acc_l2_distance / num_batch_eval
                        eval_summary_str = sess.run(eval_summary_total, feed_dict={loss_ph: eval_loss,
                                                                                   similarity_ph: eval_similarity,
                                                                                   l2_distance_ph: eval_l2_distance})
                    writer_eval.add_summary(eval_summary_str, step)

                if step % 100 == 0 or step % num_batch == 0 or step == 1:
                    print("Approximating layer: %2d | Epoch: %4d/%4d | Batch: %4d/%4d | "
                          "Train loss: %2.4f | Train similarity/L2 distance: %.4f/%2.4f | "
                          "Eval loss: %2.4f | Eval similarity/L2 distance: %.4f/%2.4f" %
                          (approximated_layer, n, epochs, b, num_batch,
                           train_loss, train_similarity, train_l2_distance,
                           eval_loss, eval_similarity, eval_l2_distance))

                if step % num_batch == 0:
                    saver_approximator.save(sess,
                                            log_dir + 'approximator_{}'.format(layer_folder_name),
                                            global_step=step)

                step += 1

        writer.close()
        writer_eval.close()

        # eval
        eval_input_ids, eval_input_mask, eval_segment_ids, \
        eval_label_ids, eval_is_real_example, \
        eval_input_tokens = generate_ph_input(batch_size=num_actual_eval_examples,
                                              seq_length=seq_length,
                                              examples=eval_examples,
                                              label_list=label_list,
                                              tokenizer=tokenizer,
                                              return_tokens=True)
        eval_token_length = [len(elem) for elem in eval_input_tokens]
        ground_truths, predictions, cosine_similarity_all = sess.run([y, y_pred, similarity],
                                                                      feed_dict={input_ids_ph: eval_input_ids,
                                                                                 input_mask_ph: eval_input_mask,
                                                                                 segment_ids_ph: eval_segment_ids})
        cosine_similarity_actual_token = 0
        cosine_similarity_padding = 0
        cosine_similarity_all = 0
        num_sample_no_padding = 0
        for i in range(num_actual_eval_examples):
            if eval_token_length[i] >= seq_length:
                num_sample_no_padding += 1
                continue
            cosine_similarity_actual_token += np.mean([1 - cosine(ground_truths[i][j], predictions[i][j])
                                                       for j in range(eval_token_length[i])])
            cosine_similarity_padding += np.mean([1 - cosine(ground_truths[i][j], predictions[i][j])
                                                  for j in range(eval_token_length[i], seq_length)])
            cosine_similarity_all += np.mean([1 - cosine(ground_truths[i][j], predictions[i][j])
                                              for j in range(seq_length)])
        cosine_similarity_actual_token /= (num_actual_eval_examples - num_sample_no_padding)
        cosine_similarity_padding /= (num_actual_eval_examples - num_sample_no_padding)
        cosine_similarity_all /= (num_actual_eval_examples - num_sample_no_padding)

        print("Skip {} samples without paddings".format(num_sample_no_padding))

        with tf.io.gfile.GFile(FLAGS.output_dir + 'results.txt', 'a') as writer:
            writer.write("Approximating layer %2d: Actual: %.4f/%.4f | Pad: %.4f/%.4f | All: %.4f/%.4f\n" %
                         (approximated_layer,
                          cosine_similarity_actual_token, 1 - cosine_similarity_actual_token,
                          cosine_similarity_padding, 1 - cosine_similarity_padding,
                          cosine_similarity_all, 1 - cosine_similarity_all))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("approximator_setting")
    flags.mark_flag_as_required("approximate_part")
    tf.compat.v1.app.run()
