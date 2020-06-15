#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-small-remove-skip-connection-layer6
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=6

#　--init_checkpoint=$MODEL_DIR/bert_model.ckpt \
# --layers_cancel_skip_connection= \

python run_finetune.py \
  --task_name=MNLIMDevAsTest \
  --do_train=true \
  --do_eval=true \
  --do_pred=false \
  --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
  --load_from_finetuned=false \
  --max_seq_length=128 \
  --num_train_epochs=3 \
  --train_batch_size=64 \
  --learning_rate=2e-5 \
  --hidden_size=512 \
  --layers_cancel_skip_connection=5 \
  --output_dir=$ROOT_DIR/tmp/no_estimator/finetune/$MODEL/mnlim_subset/
