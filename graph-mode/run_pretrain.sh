#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-small-new
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=0

# --layers_cancel_skip_connection= \
# --init_checkpoint= \

python run_pretrain.py \
  --train_input_file=$ROOT_DIR/pretrain_data_train/*.tfrecord \
  --eval_input_file=$ROOT_DIR/pretrain_data_test/*.tfrecord \
  --do_train=true \
  --do_eval=false \
  --model_type=origin \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --train_batch_size=256 \
  --eval_batch_size=64 \
  --learning_rate=1e-4 \
  --num_train_steps=1000000 \
  --num_warmup_steps=10000 \
  --print_freq=1 \
  --output_dir=$ROOT_DIR/tmp/no_estimator/pretrain/$MODEL/
