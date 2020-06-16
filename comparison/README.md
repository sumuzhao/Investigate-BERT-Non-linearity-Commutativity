## Comparison to MLP & CNN

In this repository, we play with simple multi-layer perceptrons and convolutional neural networks for comparison. 
Due to easy experimental settings, we make a simple implementation without using a lot of arguments. 
We only need to set several parameters before running certain experiments. 

We define network architectures in models where you can define variants by youself. 
We implement by **Pytorch** here. 


### Training Toy MLP, Deep MLP, CNN

`
CUDA_VISIBLE_DEVICES='0' python mlp_toy_train.py
`

`
CUDA_VISIBLE_DEVICES='0' python mlp_train.py
`

`
CUDA_VISIBLE_DEVICES='0' python cnn_train.py
`

### Train linear approximator for the toy MLP model

`
CUDA_VISIBLE_DEVICES='0' python mlp_approximator.py
`

### Swapping experiment

`
CUDA_VISIBLE_DEVICES='0' python mlp_swap.py
`

`
CUDA_VISIBLE_DEVICES='0' python cnn_swap.py
`

### Shuffling experiment

`
CUDA_VISIBLE_DEVICES='0' python mlp_shuffle.py
`

`
CUDA_VISIBLE_DEVICES='0' python cnn_shuffle.py
`