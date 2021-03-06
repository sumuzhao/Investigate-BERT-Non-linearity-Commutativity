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
import comparison.models.cnn as models


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
data = 'CIFAR10'
batch_size = 32
n_epochs = 50
lr = 0.0001
lr_decay = 0.1
if data == 'MNIST':
    c = 1
elif data == 'CIFAR10':
    c = 3
else:
    raise ValueError('Must specify data')

### Conv Net model ###
net = models.ConvNet(c)
# net = models.ConvNet_LN(c)
# net = models.ConvNet_LN_SC1(c)

# net = models.ConvNet_Long(c)
# net = models.ConvNet_Long_LN(c)
# net = models.ConvNet_Long_LN_SC(c)

# net = models.ConvNet_LN_Residual(c)
# net = models.LeNet()

# model name for experiments
model_name = "{}_data{}_bs{}_epoch{}_adam_lr{}_lrdecay{}".format(net.__class__.__name__, data, batch_size, n_epochs,
                                                                 lr, lr_decay)
print("Saved model name:", model_name)

resume_net = None
# resume_net = 'trained_convnet/{}/net_50.t7'.format(model_name)
print("Resumed model checkpoint:", resume_net)
#############################

# Load data
if data == 'MNIST':
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
elif data == 'CIFAR10':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
else:
    raise ValueError("data must be specified. ")
print("Current data {} trainset and testset:".format(data), len(trainset), len(testset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# initialize weights
if resume_net:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(resume_net)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
else:
    print('==> Random initialization..')
    init_params(net)
print(net)

# specify loss function
criterion = nn.CrossEntropyLoss()

# use gpu
print('Current devices: ' + str(torch.cuda.current_device()))
print('Device count: ' + str(torch.cuda.device_count()))
net.cuda()
criterion = criterion.cuda()

# specify optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(n_epochs):

    net.train()  # prep model for training
    # monitor training loss
    train_loss = 0.0
    correct = 0
    total = 0

    ###################
    # train the model #
    ###################
    for data, target in trainloader:
        total += data.size(0)
        # use gpu
        data = data.cuda()
        target = target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output.data, 1)
        # compare predictions to true label
        correct += pred.eq(target.data).cpu().sum().item()

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / total
    train_acc = correct / total

    net.eval()  # prep model for *evaluation*
    test_loss = 0.0
    correct = 0
    total = 0
    for data, target in testloader:
        total += data.size(0)
        # use gpu
        data = data.cuda()
        target = target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output.data, 1)
        # compare predictions to true label
        correct += pred.eq(target.data).cpu().sum().item()

    test_loss = test_loss / total
    test_acc = correct / total

    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} '
          '\tTesting Loss: {:.6f} \tTesting Accuracy: {:.6f}'.format(
        epoch + 1, train_loss, train_acc, test_loss, test_acc))

    # learning rate decay
    if epoch + 1 in [40, 80]:
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

    # store the model
    if (epoch + 1) % 10 == 0:
        state = {'state_dict': net.state_dict()}
        opt_state = {'optimizer': optimizer.state_dict()}
        if not os.path.exists('trained_convnet/{}/'.format(model_name)):
            os.makedirs('trained_convnet/{}/'.format(model_name))
        torch.save(state, 'trained_convnet/{}/'.format(model_name) + 'net_' + str(epoch + 1) + '.t7')
        torch.save(opt_state, 'trained_convnet/{}/'.format(model_name) + 'net_opt_state_' + str(epoch + 1) + '.t7')
