#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-base-uncased
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=4

important_head_per_layer=(5 1 0 1 1 8 0 8 4 0 10 2)

for j in `seq 0 11`
do
  python run_leave_one_head_per_layer_ft.py \
    --task_name=MNLIMDevAsTest \
    --do_train=true \
    --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
    --vocab_file=$MODEL_DIR/vocab.txt \
    --bert_config_file=$MODEL_DIR/bert_config.json \
    --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
    --load_from_finetuned=false \
    --max_seq_length=128 \
    --num_train_epochs=3 \
    --train_batch_size=64 \
    --learning_rate=2e-5 \
    --hidden_size=768 \
    --cur_layer=$j \
    --most_important_head=${important_head_per_layer[$j]} \
    --output_dir=$ROOT_DIR/tmp/mask_head/leave_one_head_per_layer_ft/$MODEL/mnlim_subset_exhaustive_search/
done