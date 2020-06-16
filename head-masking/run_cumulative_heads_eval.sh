#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-base-uncased
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=0


python run_cumulative_heads_eval.py \
  --task_name=MNLIM \
  --do_eval=true \
  --data_dir=$ROOT_DIR/GLUE/MNLI \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --num_train_epochs=3 \
  --train_batch_size=64 \
  --learning_rate=2e-5 \
  --hidden_size=768 \
  --importance_setting=l2_norm \
  --from_most=false \
  --output_dir=$ROOT_DIR/tmp/mask_head/cumulative_heads_eval/$MODEL/mnlim_from_least_l2_norm/
