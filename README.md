# Investigate BERT on Non-linearity and Layer Commutativity

In this repository, we implemented some empirical experiments based on the [official BERT code](https://github.com/google-research/bert). 

Paper refers to [Of Non-Linearity and Commutativity in BERT](https://drive.google.com/file/d/1ecEKvRzaSlW3lAEnItOaTqcGEobrbeFc/view?usp=sharing).

## Get started

Code Structure:
- modeling*, optimization*, tokenization. Define BERT structures, optimizer, tokenizer. 
- run_pretraining*, run_classifier. Pre-training and fine-tuning on GLUE tasks. 
- create_pretraining_data. Create pre-processed data for pre-training (Unlabeled large corpus --> TFRecord files). 
- run_finetune_glue.sh. Script for fine-tuning BERT on all GLUE tasks. 
- data_utils. Data processor for GLUE fine-tuning. 
- **graph-mode**. Refactorize run_pre-training.py and run_classifier.py in graph mode instead of using Tensorflow Estimator API
- [**non-linearity**](non-linearity/). Experiments including training linear/non-linear approximators, replacing, removing, freezing and extracting hidden embeddings. 
- [**layer-commutativity**](layer-commutativity/). Experiments including swapping and shuffling
- [**comparison**](comparison/). Compare with simple MLP and CNN models. 

Main dependencies:
- Python 3.7.5
- Tensorflow 1.１4.0
- Pytorch 1.5.0

Others refer to the requirements file. 

## Pre-training & Fine-tuning

In this work, we mainly experiment on BERT-base and BERT-small. 
For BERT-base, we use the pre-trained weights provided by the official. 
For BERT-small, we pre-trained by ourselves, but you can also use the official pre-trained weights. 

| Model           | Layer | Head | Hidden Size | Max Seq Length | #Params | Pre-trained Weights |
|------------|-------|------|-------------|----------------|---------|---------------------|
| BERT-base  | 12    | 12   | 768         | 512            | 110M    | [Official](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)                    |
| BERT-small | 6     | 8    | 512         | 128            | 35M     | [Official](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-512_A-8.zip)                    |

BERT official team did not release the pre-processed data for pre-training, and the corpora they used like English Wikipedia and Book Corpus are not available. 
So we use the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) to pre-train the BERT-small, which is composed of 38GB text. 
Note that, it took around **8** days to pre-train the BERT-small on a single Nvidia Tesla V100 GPU card and **5** days on two cards. 

Converting the original unlabeded corpus to TFRecord files is both time- and resource- comsuming. 
We recommend that you directly use the pre-trained weights. 
But if you want to use your own text to pre-train BERT, you can use following commands: 

To save your time, we provided our pre-trained weights of BERT-small. 
[Download](https://drive.google.com/file/d/1Ehld3iwF9tJMmiFvTbciTO06SXMQO4vq/view?usp=sharing)

Here are some commands for pre-training. 
Create pretraining data
```bash
python create_pretraining_data.py \
  --input_file=$RAW_TEXT_DIR \
  --output_file=$TFRECORD_DIR \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --do_lower_case=true \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

Pre-training using Estimator API
single GPU card
```bash
CUDA_VISIBLE_DEVICES="0" python run_pretraining.py \
  --input_file=$TFRECORD_DIR/*.tfrecord \
  --output_dir=$MODEL_DIR \
  --do_train=true \
  --do_eval=true \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --train_batch_size=256 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --num_warmup_steps=10000 \
  --learning_rate=1e-4 \
  --model_type=origin
```

Multiple GPU cards, implemented using hovorod
```bash
CUDA_VISIBLE_DEVICES="2,3" mpirun -np 2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x HOROVOD_TIMELINE=/disco-computing/NLP_data/tmp/pretraining/BERT-small_mg_hvd_linear_attention/timeline.json \
    -mca pml ob1 -mca btl ^openib \
    python run_pretraining_hvd.py \
    --input_file=$TFRECORD_DIR/*.tfrecord \
    --output_dir=$MODEL_DIR \
    --do_train=true \
    --do_eval=true \
    --bert_config_file=$MODEL_DIR/bert_config.json \
    --train_batch_size=256 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=1000000 \
    --num_warmup_steps=10000 \
    --learning_rate=1e-4  
```

Pre-training using refactorized version, on a single GPU card
```bash
cd graph-mode/
bash run_pretrain.sh
```

In this work, we use [GLUE benchmark](https://gluebenchmark.com/) for fine-tuning. 
Here we have several methods. 

Fine-tune on a GLUE task, e.g. MNLI-matched.
Use Estimator API. 
```bash
CUDA_VISIBLE_DEVICES="0"　python run_classifier.py \
    --task_name=MNLIM  \
    --do_train=true  \
    --do_eval=true  \
    --do_predict=true  \
    --data_dir=$GLUE_DIR/MNLI  \
    --vocab_file=$MODEL_DIR/vocab.txt  \
    --bert_config_file=$MODEL_DIR/bert_config.json \
    --init_checkpoint=$MODEL_DIR/bert_model.ckpt  \
    --max_seq_length=128  \
    --train_batch_size=32  \
    --learning_rate=2e-5  \
    --num_train_epochs=3.0  \
    --output_dir=$MODEL_Finetune_DIR/mnlim_output/
```

Use refactorized version (Need to modify the parameters inside the bash script)
```bash
cd graph-mode/
bash run_finetune.sh
```

Fine-tune all GLUE tasks. 
```bash
bash run_finetune_glue.sh
```

For easily reading the fine-tuning results. 
```bash
python glue_result.reader.py --dir=$MODEL_Finetune_DIR
```
