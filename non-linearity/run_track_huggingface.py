import torch
from transformers import *
import numpy as np
import joblib
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# load the dev set of the subset of MNLI-m
# data_handler = open('/disco-computing/NLP_data/GLUE/MNLI_subset_0.1/dev_matched.tsv', 'r')
# data_list = []
# for i, line in enumerate(data_handler.readlines()):
#     if i == 0: continue
#     data_list.append(tuple(line.split('\t')[8:10]))
# data_handler.close()
# print(len(data_list))

# load the SemEval 2012-2016, randomly select 1K samples.
data_handler = open('/disco-computing/NLP_data/SemEval/SemEval_2012-2016_all.tsv', 'r')
data_all = data_handler.readlines()
data_handler.close()
np.random.seed(0)
random_choice_idx = np.random.choice(len(data_all) - 1, 1000, replace=False)
data_list = [tuple(data_all[idx].split('\t')[1:]) for idx in random_choice_idx]
print(len(data_list))

input = tokenizer.batch_encode_plus(data_list, add_special_tokens=True, return_tensors='pt',
                                    max_length=128, pad_to_max_length=True)
# sentence_decode = [tokenizer.decode(ids).replace('[PAD]', '').strip().split(' ') for ids in input['input_ids']]
sentence_decode = [[token for token in tokenizer.convert_ids_to_tokens(ids) if token != '[PAD]'] for ids in input['input_ids']]
# print(sentence_decode[0])
print(len(sentence_decode))
print(input['input_ids'].shape)

with torch.no_grad():
    output = model(input['input_ids'], token_type_ids=input['token_type_ids'])[2]
print([t.shape for t in output])

output_dict = {"layer_output": np.zeros((12, len(data_list), 128, 768))}
output_dict["final_embeddings"] = output[0].numpy()
for i, layer in enumerate(output[1:]):
    output_dict['layer_output'][i] = layer.numpy()
output_dict["sentence"] = sentence_decode
print([type(output_dict[key]) for key in output_dict])

save_folder = '/disco-computing/NLP_data/tmp/track_huggingface/1000_examples/BERT-base-uncased/semeval_subset_nonft/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
joblib.dump(output_dict, save_folder + 'track_embeddings.pickle')
