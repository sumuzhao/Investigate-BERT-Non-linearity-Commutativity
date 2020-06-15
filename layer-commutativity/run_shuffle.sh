#!/bin/sh

# for suzhao account
export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-small-remove-skip-connection-layer6
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export CUDA_VISIBLE_DEVICES=6


# --layers_cancel_skip_connection= \

#max=9
#
#for j in `seq 0 $max`
#do
#
#  echo "$j run! "
#
#  python run_shuffle.py \
#    --task_name=MNLIMDevAsTest \
#    --do_train=true \
#    --do_eval=true \
#    --do_pred=false \
#    --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
#    --vocab_file=$MODEL_DIR/vocab.txt \
#    --bert_config_file=$MODEL_DIR/bert_config.json \
#    --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
#    --load_from_finetuned=false \
#    --max_seq_length=128 \
#    --num_train_epochs=3 \
#    --train_batch_size=64 \
#    --learning_rate=2e-5 \
#    --hidden_size=768 \
#    --shuffle_setting=fix1-4 \
#    --freeze_part=allbut1-4,9-12_idx \
#    --np_seed=$j \
#    --output_dir=$ROOT_DIR/tmp/shuffle/$MODEL/mnlim_subset_fix1-4_freeze_allbut1-4,9-12_idx/
#
#done


# run inference
#path=mnlim_subset_fix1-4_5-8,9-12_freeze_nothing
#for dir in $(ls -l $ROOT_DIR/tmp/shuffle_new/$MODEL/$path |grep "^d" |awk '{print $9}')
#do
#  echo $dir
#  file=`ls $ROOT_DIR/tmp/shuffle_new/$MODEL/$path/$dir/ | grep "shuffle_" | grep ".meta"`
#  filename=`echo "$file" | sed s/.meta//`
#  echo $filename
#  order=`echo "$filename" | sed s/shuffle_// | sed s/-1842//`
#  echo $order
#
#  python run_shuffle.py \
#    --task_name=MNLIMDevAsTest \
#    --do_train=false \
#    --do_eval=true \
#    --do_pred=false \
#    --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
#    --vocab_file=$MODEL_DIR/vocab.txt \
#    --bert_config_file=$MODEL_DIR/bert_config.json \
#    --init_checkpoint=$ROOT_DIR/tmp/shuffle_new/$MODEL/$path/$dir/$filename \
#    --load_from_finetuned=true \
#    --max_seq_length=128 \
#    --hidden_size=768 \
#    --layers=$order \
#    --output_dir=$ROOT_DIR/tmp/shuffle_new/$MODEL/$path/
#
#done


#for j in `seq 0 -1 0`
#do
#
#  echo "Current fix $j layers. "
#
#  for line in `cat shuffle_exhaustive_search/fix${j}_layers.txt`
#  do
#
#    layer_order=$line
#    echo "Current shuffle layer order is $layer_order"
#
#    python run_shuffle.py \
#      --task_name=MNLIMDevAsTest \
#      --do_train=true \
#      --do_eval=true \
#      --do_pred=false \
#      --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
#      --vocab_file=$MODEL_DIR/vocab.txt \
#      --bert_config_file=$MODEL_DIR/bert_config.json \
#      --init_checkpoint=$ROOT_DIR/tmp/origin/$MODEL/mnlim_subset/origin-1842 \
#      --load_from_finetuned=true \
#      --max_seq_length=128 \
#      --num_train_epochs=3 \
#      --train_batch_size=64 \
#      --learning_rate=2e-5 \
#      --hidden_size=768 \
#      --layers=$layer_order \
#      --freeze_part=nothing \
#      --output_dir=$ROOT_DIR/tmp/shuffle/shuffle_exhaustive_search_continue_train/$MODEL/mnlim_subset_fix${j}layers_freeze_nothing/
#
#  done
#
#done

max=5
for j in `seq 0 $max`
do

  echo "Current layer $j"

  python run_shuffle.py \
    --task_name=MNLIMDevAsTest \
    --do_train=true \
    --do_eval=true \
    --do_pred=false \
    --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
    --vocab_file=$MODEL_DIR/vocab.txt \
    --bert_config_file=$MODEL_DIR/bert_config.json \
    --init_checkpoint=$ROOT_DIR/tmp/no_estimator/finetune/$MODEL/mnlim_subset/origin-1842 \
    --load_from_finetuned=true \
    --max_seq_length=128 \
    --num_train_epochs=3 \
    --train_batch_size=64 \
    --learning_rate=2e-5 \
    --hidden_size=512 \
    --layers=$j \
    --shuffle_setting=albert-like \
    --freeze_part=nothing \
    --layers_cancel_skip_connection=5 \
    --output_dir=$ROOT_DIR/tmp/shuffle/albert-like_continue_train/$MODEL/mnlim_subset_repeat_layer_freeze_nothing/

done

#python run_shuffle.py \
#  --task_name=MNLIMDevAsTest \
#  --do_train=true \
#  --do_eval=true \
#  --do_pred=false \
#  --data_dir=$ROOT_DIR/GLUE/MNLI_subset_0.1 \
#  --vocab_file=$MODEL_DIR/vocab.txt \
#  --bert_config_file=$MODEL_DIR/bert_config.json \
#  --init_checkpoint=$ROOT_DIR/tmp/origin/$MODEL/mnlim_subset/origin-1842 \
#  --load_from_finetuned=true \
#  --max_seq_length=128 \
#  --num_train_epochs=3 \
#  --train_batch_size=64 \
#  --learning_rate=2e-5 \
#  --hidden_size=768 \
#  --layers=0,1,2,3,0,1,2,3,0,1,2,3 \
#  --freeze_part=nothing \
#  --output_dir=$ROOT_DIR/tmp/shuffle/albert-like_continue_train/$MODEL/mnlim_subset_repeat_freeze_nothing/