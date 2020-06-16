#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-base-uncased
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=6


# --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
# --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
# --layers_cancel_skip_connection= \
# --layers_use_relu= \

python run_track.py \
  --task_name=MNLIMDevAsTest \
  --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --hidden_size=768 \
  --n_layers=12 \
  --feed_ones=false \
  --feed_same=false \
  --output_dir=$ROOT_DIR/tmp/track/all_examples/$MODEL/mnlim_subset_nonft/