#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-base-uncased
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=5

echo $MODEL

max=11

for j in `seq 0 $max`
do

  # --layers_cancel_skip_connection= \
  # --layers_use_relu= \

  python run_train.py \
     --task_name=MNLIMDevAsTest \
     --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
     --vocab_file=$MODEL_DIR/vocab.txt \
     --bert_config_file=$MODEL_DIR/bert_config.json \
     --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
     --max_seq_length=128 \
     --num_train_epochs=3 \
     --learning_rate=1e-3 \
     --loss=cosine \
     --train_batch_size=64 \
     --hidden_size=512 \
     --layers=$j \
     --approximate_part=attention \
     --approximator_setting=HS_Attention \
     --output_dir=$ROOT_DIR/tmp/approximator/$MODEL/mnlim_subset_nonft_cosine_hs_attention/

done

#for j in `seq 0 $max`
#do
#
#  # --layers_cancel_skip_connection= \
#  # --layers_use_relu= \
#
#  python run_approximator.py \
#     --task_name=MNLIMDevAsTest \
#     --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
#     --vocab_file=$MODEL_DIR/vocab.txt \
#     --bert_config_file=$MODEL_DIR/bert_config.json \
#     --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
#     --max_seq_length=128 \
#     --num_train_epochs=3 \
#     --learning_rate=1e-3 \
#     --loss=cosine \
#     --train_batch_size=64 \
#     --hidden_size=512 \
#     --layers=$j \
#     --approximate_part=self_attention_ff \
#     --approximator_setting=HS_Self_Attention_FF \
#     --layers_use_relu=2,4 \
#     --output_dir=$ROOT_DIR/tmp/non-linearity/$MODEL/mnlim_subset_nonft_cosine_hs_self_attention_ff/
#
#done

#for j in `seq 0 $max`
#do
#
#  # --layers_cancel_skip_connection= \
#  # --layers_use_relu= \
#
#  python run_approximator.py \
#     --task_name=MNLIMDevAsTest \
#     --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
#     --vocab_file=$MODEL_DIR/vocab.txt \
#     --bert_config_file=$MODEL_DIR/bert_config.json \
#     --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
#     --max_seq_length=128 \
#     --num_train_epochs=3 \
#     --learning_rate=1e-3 \
#     --loss=cosine \
#     --train_batch_size=64 \
#     --hidden_size=768 \
#     --layers=$j \
#     --use_nonlinear_approximator=true \
#     --use_dropout=false \
#     --approximate_part=self_attention_ff \
#     --approximator_setting=HS_Self_Attention_FF \
#     --output_dir=$ROOT_DIR/tmp/non-linearity/$MODEL/mnlim_subset_nonft_cosine_nla1_hs_self_attention_ff/
#
#done
