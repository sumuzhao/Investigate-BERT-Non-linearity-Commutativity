from __future__ import print_function
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('..')
import comparison.models.mlp as models


class Approximator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Approximator, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.linear_layer(x)
        return out


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


# Set the seed for reproducing the results
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = True


#############################
# Parameters needed setting #
#############################
# training parameters
batch_size = 32
n_epochs = 30
lr = 0.001

# regularization
norm_order = 1
reg_weight = 1

# dropout
p = 0.99

# linear non-linearity
approximated_layer = 2

### MLP model ###
mlp = models.MLP()
# mlp = models.MLP_dropout_before_relu(p)
# mlp = models.MLP_dropout_after_relu(p)
# mlp = models.DeepMLP()

# model name for experiments
model_name = "{}_bs{}_epoch{}_adam_lr{}_dropout{}_l{}reg_{}_layer1".\
             format(mlp.__class__.__name__, batch_size, n_epochs, lr, p, norm_order, reg_weight)
print("Saved model name:", model_name)

resume_mlp = 'trained_mlp/{}/mlp_30.t7'.format(model_name)
print("Resumed model checkpoint:", resume_mlp)
#############################


# Load data
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainset_large, trainset_small = torch.utils.data.random_split(trainset, [50000, 10000])
print("Current trainset for mlp and la:", len(trainset_large), len(trainset_small))

trainloader_mlp = torch.utils.data.DataLoader(trainset_large, batch_size=batch_size, shuffle=True)
trainloader_la = torch.utils.data.DataLoader(trainset_small, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# Load checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load(resume_mlp)
mlp.load_state_dict(checkpoint['state_dict'])
print(mlp)

# use gpu
print('Current devices: ' + str(torch.cuda.current_device()))
print('Device count: ' + str(torch.cuda.device_count()))
mlp.cuda()


### Linear non-linearity ###
la = Approximator(28 * 28, 768)
init_params(la)
print(la)

# loss, use cosine distance as loss.
criterion_la = nn.CosineEmbeddingLoss()

# use gpu
la.cuda()
criterion_la = criterion_la.cuda()

# specify optimizer
optimizer_la = torch.optim.Adam(la.parameters(), lr=0.001)

mlp.eval()
for epoch in range(10):

    la.train()  # prep model for training
    # monitor training loss
    train_loss = 0.0
    total = 0

    ###################
    # train the la #
    ###################
    for data, target in trainloader_la:
        total += data.size(0)
        # use gpu
        data = data.cuda()
        target = target.cuda()
        # clear the gradients of all optimized variables
        optimizer_la.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        _, extract_embeddings = mlp(data)
        x, y = extract_embeddings[0], extract_embeddings[approximated_layer]
        y_pred = la(x)
        # calculate the loss
        loss = criterion_la(y, y_pred, torch.ones(1).cuda())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer_la.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / total

    la.eval()  # prep model for *evaluation*
    test_loss = 0.0
    total = 0
    for data, target in testloader:
        total += data.size(0)
        # use gpu
        data = data.cuda()
        target = target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        _, extract_embeddings = mlp(data)
        x, y = extract_embeddings[0], extract_embeddings[approximated_layer]
        y_pred = la(x)
        # calculate the loss
        loss = criterion_la(y, y_pred, torch.ones(1).cuda())
        # update test loss
        test_loss += loss.item() * data.size(0)

    test_loss = test_loss / total

    print('Epoch: {} \tTraining cosine distance: {:.6f} \tTesting cosine distance: {:.6f}'.format(
        epoch + 1, train_loss, test_loss))
