#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-base-uncased
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=6

# for multi_layer input
layers=(0 1 2 3 4 5 6 7 8 9 10 11)
max=11

#for j in `seq 0 11`
#do

  # 0 -- 0,1 -- 0,1,2 -- ...
#  multi_layer=${layers[@]:0:$j + 1}

  # 11 -- 10,11 -- 9,10,11 -- ...
#  multi_layer=${layers[@]:11 - $j}

# $j
# ${multi_layer// /,} \

#  echo "$j layer"
#  echo $multi_layer

  for i in `seq 0 4`
  do

  echo "$i run"

  python run_freeze.py \
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
    --hidden_size=768 \
    --layers=-1 \
    --tf_seed=$i \
    --freeze_part=allbutpooler+output \
    --output_dir=$ROOT_DIR/tmp/freeze/$MODEL/mnlim_subset_freeze_allbutpooler+output_5runs/

  done

#done