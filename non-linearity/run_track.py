"""Track the hidden token embeddings through the model"""

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
import joblib
import os
from tqdm import tqdm

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
                        MnliMDevAsTestProcessor,
                        SemEvalProcessor)
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

flags.DEFINE_bool("feed_ones", False, "Whether to feed ones for each layer.")

flags.DEFINE_bool("feed_same", False, "Whether to feed ones for each layer.")

flags.DEFINE_integer("n_layers", 12, "Number of layers, 12 for base, 6 for small")

flags.DEFINE_string("layers_cancel_skip_connection", None, "layers need to cancel skip-connection")

flags.DEFINE_string("layers_use_relu", None, "layers need to use_ReLU")


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
        "mnlimdevastest": MnliMDevAsTestProcessor,
        "semeval": SemEvalProcessor
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

    label_list = processor.get_labels()
    print("Label list of current task", label_list)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    print("num_actual_eval_examples", num_actual_eval_examples)

    n_layers = FLAGS.n_layers
    embed_dim = FLAGS.hidden_size  # hidden size, 768 for BERT-base, 512 for BERT-small
    seq_length = FLAGS.max_seq_length

    # Define some placeholders for the input
    input_ids_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='input_ids')
    input_mask_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='input_mask')
    segment_ids_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, seq_length], name='segment_ids')

    tf.compat.v1.logging.info("Running tracking hidden token embeddings through the model!")

    # for faster calculation
    # np.random.seed(0)
    # random_choice_idx = np.random.choice(num_actual_eval_examples, 1000, replace=False)
    # eval_examples = [eval_examples[idx] for idx in random_choice_idx]
    # num_actual_eval_examples = 1000
    if FLAGS.feed_ones:
        num_actual_eval_examples = 1
    print("here we only take {} samples.".format(num_actual_eval_examples))
    tf.compat.v1.logging.info("For faster calculation, we reduce the example size! ")

    save_file = 'track_embeddings.pickle'
    # output = {"word_embeddings": np.zeros((num_actual_eval_examples, seq_length, embed_dim)),
    #           "final_embeddings": np.zeros((num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_input": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_self_attention": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_self_attention_ff": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_attention_before_ln": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_attention": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_ffgelu": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim * 4)),
    #           "layer_mlp": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_ffn_before_ln": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "layer_output": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
    #           "sentence": []}
    output = {"final_embeddings": np.zeros((num_actual_eval_examples, seq_length, embed_dim)),
              "layer_self_attention_ff": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
              "layer_attention": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
              "layer_mlp": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
              "layer_output": np.zeros((n_layers, num_actual_eval_examples, seq_length, embed_dim)),
              "sentence": []}

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
        cancel_skip_connection=layers_cancel_skip_connection,
        layer_use_relu=layers_use_relu,
        feed_same=FLAGS.feed_same)

    # extract all the intermediate outputs
    # word_embeddings = model.get_in_embeds()[0]
    final_embeddings = model.get_embedding_output()
    # layer_self_attention = model.get_all_head_output()
    layer_self_attention_ff = model.get_all_attention_before_dropout()
    # layer_attention_before_ln = model.get_all_attention_before_layernorm()
    layer_attention = model.get_all_layer_tokens_beforeMLP()
    # layer_ffgelu = model.get_all_layer_tokens_after_FFGeLU()
    layer_mlp = model.get_all_layer_tokens_afterMLP()
    # layer_ffn_before_ln = model.get_all_ffn_before_layernorm()
    layer_output = model.get_all_encoder_layers()

    # load weights
    init_checkpoint = FLAGS.init_checkpoint
    tvars = tf.compat.v1.trainable_variables()
    saver_init = tf.train.Saver(tvars)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver_init.restore(sess, init_checkpoint)

        if FLAGS.feed_ones:
            input_ids = np.zeros([1, seq_length])
            input_mask = np.ones([1, seq_length])
            segment_ids = np.zeros([1, seq_length])
        else:
            input_ids, input_mask, segment_ids, \
            label_ids, is_real_example, input_tokens = generate_ph_input(batch_size=num_actual_eval_examples,
                                                                         seq_length=seq_length,
                                                                         examples=eval_examples,
                                                                         label_list=label_list,
                                                                         tokenizer=tokenizer,
                                                                         return_tokens=True)

        # word_embeddings_val, final_embeddings_val, \
        # layer_self_attention_val, layer_self_attention_ff_val, \
        # layer_attention_before_ln_val, layer_attention_val, \
        # layer_ffgelu_val, layer_mlp_val, layer_ffn_before_ln_val, \
        # layer_output_val = sess.run([word_embeddings, final_embeddings,
        #                              layer_self_attention, layer_self_attention_ff,
        #                              layer_attention_before_ln, layer_attention,
        #                              layer_ffgelu, layer_mlp, layer_ffn_before_ln, layer_output],
        #                             feed_dict={input_ids_ph: input_ids,
        #                                        input_mask_ph: input_mask,
        #                                        segment_ids_ph: segment_ids})

        final_embeddings_val, \
        layer_self_attention_ff_val, layer_attention_val, layer_mlp_val, \
        layer_output_val = sess.run([final_embeddings,
                                     layer_self_attention_ff,
                                     layer_attention,
                                     layer_mlp, layer_output],
                                    feed_dict={input_ids_ph: input_ids,
                                               input_mask_ph: input_mask,
                                               segment_ids_ph: segment_ids})

        # assign values to output dict
        # output["word_embeddings"] = np.reshape(word_embeddings_val,
        #                                        [num_actual_eval_examples, seq_length, embed_dim])
        output["final_embeddings"] = np.reshape(final_embeddings_val,
                                                [num_actual_eval_examples, seq_length, embed_dim])
        for layer in tqdm(range(n_layers)):
            # if FLAGS.feed_same:
            #     output["layer_input"][layer] = np.reshape(final_embeddings_val,
            #                                               [num_actual_eval_examples, seq_length, embed_dim])
            # else:
            #     if layer == 0:
            #         output["layer_input"][layer] = np.reshape(final_embeddings_val,
            #                                                   [num_actual_eval_examples, seq_length, embed_dim])
            #     else:
            #         output["layer_input"][layer] = np.reshape(layer_output_val[layer - 1],
            #                                                   [num_actual_eval_examples, seq_length, embed_dim])
            # output["layer_self_attention"][layer] = np.reshape(layer_self_attention_val[layer],
            #                                                    [num_actual_eval_examples, seq_length, embed_dim])
            output["layer_self_attention_ff"][layer] = np.reshape(layer_self_attention_ff_val[layer],
                                                                  [num_actual_eval_examples, seq_length, embed_dim])
            # output["layer_attention_before_ln"][layer] = np.reshape(layer_attention_before_ln_val[layer],
            #                                                         [num_actual_eval_examples, seq_length, embed_dim])
            output["layer_attention"][layer] = np.reshape(layer_attention_val[layer],
                                                          [num_actual_eval_examples, seq_length, embed_dim])
            # output["layer_ffgelu"][layer] = np.reshape(layer_ffgelu_val[layer],
            #                                            [num_actual_eval_examples, seq_length, embed_dim * 4])
            output["layer_mlp"][layer] = np.reshape(layer_mlp_val[layer],
                                                    [num_actual_eval_examples, seq_length, embed_dim])
            # output["layer_ffn_before_ln"][layer] = np.reshape(layer_ffn_before_ln_val[layer],
            #                                                   [num_actual_eval_examples, seq_length, embed_dim])
            output["layer_output"][layer] = np.reshape(layer_output_val[layer],
                                                       [num_actual_eval_examples, seq_length, embed_dim])
        if FLAGS.feed_ones:
            output["sentence"] = [[''] * seq_length]
        else:
            output["sentence"] = input_tokens

        joblib.dump(output, os.path.join(FLAGS.output_dir, save_file))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
