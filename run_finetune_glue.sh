#!/bin/sh

# run all GLUE tasks, training, evaluation, prediction.
# Package the predictions into GLUE submission files.

export ROOT_DIR=/disco-computing/NLP_data
export MODEL=BERT-small-add-GeLU-attention
export MODEL_DIR=$ROOT_DIR/BERT-pretrained-model/$MODEL
export OUTPUT_DIR=$ROOT_DIR/tmp/finetune/$MODEL
export CUDA_VISIBLE_DEVICES=0

function run_task() {
  echo "Current task is $1 | batch size: $3 | learning rate: $4 | train and warmup steps: $5, $6."
  python run_classifier.py   \
      --task_name=$1   \
      --do_train=true   \
      --do_eval=true   \
      --do_predict=true \
      --data_dir=$ROOT_DIR/GLUE/$2   \
      --vocab_file=$MODEL_DIR/vocab.txt   \
      --bert_config_file=$MODEL_DIR/bert_config.json   \
      --init_checkpoint=$MODEL_DIR/bert_model.ckpt   \
      --max_seq_length=128   \
      --train_batch_size=$3   \
      --learning_rate=$4   \
      --train_step=$5   \
      --warmup_step=$6   \
      --output_dir=$OUTPUT_DIR   \
      --add_GeLU_att=$add_GeLU_att \
      --add_weight=$add_weight   \
      --weight_type=$weight_type   \
      --weight_activation=$weight_activation   \
      --linear_attention=$linear_attention   \
      --model_type=$model_type   \
      --layers_cancel_skip_connection=$layers_cancel_skip_connection   \
      --layers_use_relu=$layers_use_relu
}

function run_ax() {
  echo "Current task is ax."
  python run_classifier.py   \
      --task_name=ax   \
      --do_predict=true \
      --data_dir=$ROOT_DIR/GLUE/AX   \
      --vocab_file=$MODEL_DIR/vocab.txt   \
      --bert_config_file=$MODEL_DIR/bert_config.json   \
      --init_checkpoint=$OUTPUT_DIR/mnlim_output/model.ckpt-10000   \
      --max_seq_length=128   \
      --output_dir=$OUTPUT_DIR   \
      --add_GeLU_att=$add_GeLU_att \
      --add_weight=$add_weight   \
      --weight_type=$weight_type   \
      --weight_activation=$weight_activation   \
      --linear_attention=$linear_attention   \
      --model_type=$model_type   \
      --layers_cancel_skip_connection=$layers_cancel_skip_connection   \
      --layers_use_relu=$layers_use_relu
}


case $MODEL in
BERT-base-uncase)
    echo "Model is BERT-base-uncased!"
    export model_type=origin
    export linear_attention=false

    # for MNLI and QQP, batch size 128 will cause OOM
    # so change it as 96 and increase the training steps
    run_task cola CoLA 16 1e-5 5336 320
    run_task mnlim MNLI 96 3e-5 12000 1200
    run_task mnlimm MNLI 96 3e-5 12000 1200
    run_task mrpc MRPC 32 2e-5 800 200
    run_task qnli QNLI 32 1e-5 33112 1986
    run_task qqp QQP 96 5e-5 19000 1900
    run_task rte RTE 32 3e-5 800 200
    run_task sst2 SST-2 32 1e-5 20935 1256
    run_task stsb STS-B 16 2e-5 3598 214
    run_task wnli WNLI 8 2e-5 800 200
    run_ax

    exit 0
    ;;
BERT-small)
    echo "Model is BERT-small (two card, 512 batch size)!"
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-new)
    echo "Model is BERT-small-new (one card, 256 batch size)!"
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-new-1.5M)
    echo "Model is BERT-small-new-1.5M (one card, 256 batch size, 1.5M steps)!"
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-new-2M)
    echo "Model is BERT-small-new-2M (one card, 256 batch size, 2M steps)!"
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-weighted-ffn)
    echo "Model is BERT-small-weighted-ffn!"
    export add_weight=ffn
    export weight_type=learn
    export weight_activation=sigmoid
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-weighted-attention)
    echo "Model is BERT-small-weighted-attention!"
    export add_weight=attention
    export weight_type=learn
    export weight_activation=sigmoid
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-weighted-all)
    echo "Model is BERT-small-weighted-all!"
    export add_weight=all
    export weight_type=learn
    export weight_activation=sigmoid
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-weighted-attention-wo-sigmoid)
    echo "Model is BERT-small-weighted-attention-wo-sigmoid!"
    export add_weight=attention
    export weight_type=learn
    export weight_activation=linear
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-weighted-attention-static)
    echo "Model is BERT-small-weighted-attention-static!"
    export add_weight=attention
    export weight_type=static
    export weight_activation=linear
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-linear-attention)
    echo "BERT-small-linear-attention!"
    export linear_attention=true
    export model_type=origin
    ;;
BERT-small-linear-FFN)
    echo "BERT-small-linear-FFN!"
    export linear_attention=false
    export model_type=origin
    ;;
BERT-small-add-GeLU-attention)
    echo "BERT-small-add-GeLU-attention!"
    export model_type=origin
    export add_GeLU_att=true
    export linear_attention=false
    ;;
BERT-small-wo-FFN-increase-attention)
    echo "BERT-small-wo-FFN-increase-attention!"
    export model_type=no-ffn
    export linear_attention=false
    ;;
BERT-small-wo-FFN-add-GeLU-attention)
    echo "BERT-small-wo-FFN-add-GeLU-attention!"
    export model_type=no-ffn
    export linear_attention=false
    ;;
BERT-small-wo-FFN)
    echo "BERT-small-wo-FFN!"
    export model_type=no-ffn
    export linear_attention=false
    ;;
BERT-small-wo-FFN-increase-add-GeLU-attention)
    echo "BERT-small-wo-FFN-increase-add-GeLU-attention!"
    export model_type=no-ffn
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer1)
    echo "BERT-small-remove-skip-connection-layer1!"
    export model_type=origin
    export layers_cancel_skip_connection=0
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer1-1.5M)
    echo "BERT-small-remove-skip-connection-layer1-1.5M!"
    export model_type=origin
    export layers_cancel_skip_connection=0
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer1-2M)
    echo "BERT-small-remove-skip-connection-layer1-2M!"
    export model_type=origin
    export layers_cancel_skip_connection=0
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer2)
    echo "BERT-small-remove-skip-connection-layer2!"
    export model_type=origin
    export layers_cancel_skip_connection=1
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer3)
    echo "BERT-small-remove-skip-connection-layer3!"
    export model_type=origin
    export layers_cancel_skip_connection=2
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer4)
    echo "BERT-small-remove-skip-connection-layer4!"
    export model_type=origin
    export layers_cancel_skip_connection=3
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer5)
    echo "BERT-small-remove-skip-connection-layer5!"
    export model_type=origin
    export layers_cancel_skip_connection=4
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer6)
    echo "BERT-small-remove-skip-connection-layer6!"
    export model_type=origin
    export layers_cancel_skip_connection=5
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer3,5)
    echo "BERT-small-remove-skip-connection-layer3,5!"
    export model_type=origin
    export layers_cancel_skip_connection=2,4
    export linear_attention=false
    ;;
BERT-small-remove-skip-connection-layer1,3,5)
    echo "BERT-small-remove-skip-connection-layer1,3,5!"
    export model_type=origin
    export layers_cancel_skip_connection=0,2,4
    export linear_attention=false
    ;;
BERT-small-new-layernorm)
    echo "BERT-small-new-layernorm!"
    export model_type=new-layernorm
    export linear_attention=false
    ;;
BERT-small-relu-all)
    echo "BERT-small-relu-all!"
    export model_type=origin
    export linear_attention=false
    ;;
BERT-small-relu-layer1)
    echo "BERT-small-relu-layer1!"
    export model_type=origin
    export layers_use_relu=0
    export linear_attention=false
    ;;
BERT-small-relu-layer2)
    echo "BERT-small-relu-layer2!"
    export model_type=origin
    export layers_use_relu=1
    export linear_attention=false
    ;;
BERT-small-relu-layer3)
    echo "BERT-small-relu-layer3!"
    export model_type=origin
    export layers_use_relu=2
    export linear_attention=false
    ;;
BERT-small-relu-layer4)
    echo "BERT-small-relu-layer4!"
    export model_type=origin
    export layers_use_relu=3
    export linear_attention=false
    ;;
BERT-small-relu-layer5)
    echo "BERT-small-relu-layer5!"
    export model_type=origin
    export layers_use_relu=4
    export linear_attention=false
    ;;
BERT-small-relu-layer6)
    echo "BERT-small-relu-layer6!"
    export model_type=origin
    export layers_use_relu=5
    export linear_attention=false
    ;;
BERT-small-relu-layer3,5)
    echo "BERT-small-relu-layer3,5!"
    export model_type=origin
    export layers_use_relu=2,4
    export linear_attention=false
    ;;
BERT-small-relu-layer1,3,5)
    echo "BERT-small-relu-layer1,3,5!"
    export model_type=origin
    export layers_use_relu=0,2,4
    export linear_attention=false
    ;;
*)
    echo "Please check the model name!"
    exit 0
    ;;
esac

# BERT-small
run_task cola CoLA 16 1e-5 5336 320
run_task mnlim MNLI 128 3e-5 10000 1000
run_task mnlimm MNLI 128 3e-5 10000 1000
run_task mrpc MRPC 32 2e-5 800 200
run_task qnli QNLI 32 1e-5 33112 1986
run_task qqp QQP 128 5e-5 14000 1000
run_task rte RTE 32 3e-5 800 200
run_task sst2 SST-2 32 1e-5 20935 1256
run_task stsb STS-B 16 2e-5 3598 214
run_task wnli WNLI 8 2e-5 800 200
run_ax

exit 0
