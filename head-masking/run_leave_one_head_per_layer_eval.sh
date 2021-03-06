#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-base-uncased
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=2


python run_leave_one_head_per_layer_eval.py \
  --task_name=MNLIMDevAsTest \
  --do_eval=true \
  --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=/disco-computing/NLP_data/tmp/origin/BERT-base-uncased/mnlim_subset/origin-1842 \
  --max_seq_length=128 \
  --eval_batch_size=64 \
  --hidden_size=768 \
  --importance_setting=exhaustive_search \
  --output_dir=$ROOT_DIR/tmp/mask_head/leave_one_head_per_layer_eval/$MODEL/mnlim_subset_exhaustive_search/
