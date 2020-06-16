#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-base-uncased
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=0

# for multi_layer input
#layers=(0 1 2 3 4 5 6 7 8 9 10 11)

# for BERT-base, different swapping settings
#layers=(0,1 1,2 2,3 3,4 4,5 5,6 6,7 7,8 8,9 9,10 10,11)
layers=(0,1 0,2 0,3 0,4 0,5 0,6 0,7 0,8 0,9 0,10 0,11)
#layers=(1,2 1,3 1,4 1,5 1,6 1,7 1,8 1,9 1,10 1,11)
#layers=(2,3 2,4 2,5 2,6 2,7 2,8 2,9 2,10 2,11)
#layers=(3,4 3,5 3,6 3,7 3,8 3,9 3,10 3,11)
#layers=(4,5 4,6 4,7 4,8 4,9 4,10 4,11)
#layers=(5,6 5,7 5,8 5,9 5,10 5,11)
#layers=(6,7 6,8 6,9 6,10 6,11)
#layers=(7,8 7,9 7,10 7,11)
#layers=(8,9 8,10 8,11)
#layers=(9,10 9,11)
#layers=(10,11)
#layers=(0,11 1,10 2,9 3,8 4,7 5,6)
#layers=(0,1,2 1,2,3 2,3,4 3,4,5 4,5,6 5,6,7 6,7,8 7,8,9 8,9,10 9,10,11)

# for BERT-small
#layers=(0,1 1,2 2,3 3,4 4,5)
#layers=(1,2 1,3 1,4 1,5)
#layers=(2,3 2,4 2,5)
#layers=(3,4 3,5)
#layers=(4,5)
#layers=(0,1 0,2 0,3 0,4 0,5 1,2 1,3 1,4 1,5 2,3 2,4 2,5 3,4 3,5 4,5)
#layers=(0,1,2 1,2,3 2,3,4 3,4,5)


max=10

for j in `seq 0 $max`
do

  # 0 -- 0,1 -- 0,1,2 -- ...
#  multi_layer=${layers[@]:0:$j + 1}

  # 11 -- 10,11 -- 9,10,11 -- ...
#  multi_layer=${layers[@]:11 - $j}

# $j
# ${multi_layer// /,} \
  multi_layer=${layers[$j]}

#  echo $j
  echo $multi_layer

#  --init_checkpoint=$MODEL_DIR/bert_model.ckpt \

  python run_swap.py \
    --task_name=MNLIMDevAsTest \
    --do_train=false \
    --do_eval=true \
    --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
    --vocab_file=$MODEL_DIR/vocab.txt \
    --bert_config_file=$MODEL_DIR/bert_config.json \
    --init_checkpoint=$ROOT_DIR/tmp/origin/$MODEL/mnlim_subset/origin-1842 \
    --load_from_finetuned=true \
    --max_seq_length=128 \
    --num_train_epochs=3 \
    --train_batch_size=64 \
    --learning_rate=2e-5 \
    --layers=$multi_layer \
    --hidden_size=768 \
    --output_dir=$ROOT_DIR/tmp/swap/swap_eval/$MODEL/mnlim_subset_to_layer1/

done