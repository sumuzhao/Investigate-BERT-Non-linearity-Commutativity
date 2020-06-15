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
import glob
import copy
import joblib
sys.path.append('..')
import comparison.models.mlp as models


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


# load shuffling orders
shuffle_order_files = sorted(glob.glob('shuffle_exhaustive_search/*.txt'), reverse=True)
print(shuffle_order_files)

results = {k: {'Direct_eval': [], 'Score_1_epoch': [], 'Score_5_epoch': []} for k in range(9, -1, -1)}
print(results)

for i, k in enumerate(range(9, -1, -1)):
    shuffle_order_file = open(shuffle_order_files[i], 'r')
    orders = [list(map(int, line.replace('\n', '').split(','))) for line in shuffle_order_file.readlines()]

    for order in orders:

        print("Fix {} layers and current order {}".format(k, order))

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
        n_epochs = 100
        lr = 0.0001
        lr_decay = 0.1

        # shuffle layers
        do_shuffle = True
        continue_train = False
        continue_train_epochs = 5

        ### MLP model ###
        mlp = models.DeepMLP_Long()
        # mlp = models.DeepMLP_Long_LN()
        # mlp = models.DeepMLP_Long_LN_SC()

        # model name for experiments
        model_name = "{}_bs{}_epoch{}_adam_lr{}_lrdecay{}".format(mlp.__class__.__name__, batch_size, n_epochs, lr,
                                                                  lr_decay)
        print("Saved model name:", model_name)

        resume_mlp = 'trained_mlp/{}/mlp_{}.t7'.format(model_name, n_epochs)
        print("Resumed model checkpoint:", resume_mlp)
        #############################


        # Load data
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        print("Current trainset and testset:", len(trainset), len(testset))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


        if resume_mlp:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(resume_mlp)
            state_dict = checkpoint['state_dict']
            # change the shuffled layer state
            original_state_dict = copy.deepcopy(state_dict)
            for idx, layer in enumerate(order):
                state_dict['layer{}.weight'.format(idx + 1)] = original_state_dict['layer{}.weight'.format(layer)]
                state_dict['layer{}.bias'.format(idx + 1)] = original_state_dict['layer{}.bias'.format(layer)]
                if 'ln' in model_name:
                    state_dict['layernorm{}.weight'.format(idx + 1)] = original_state_dict['layernorm{}.weight'.format(layer)]
                    state_dict['layernorm{}.bias'.format(idx + 1)] = original_state_dict['layernorm{}.bias'.format(layer)]
            mlp.load_state_dict(state_dict)
        else:
            print('==> Random initialization..')
            init_params(mlp)
        print(mlp)

        # specify loss function
        criterion = nn.CrossEntropyLoss()

        # use gpu
        print('Current devices: ' + str(torch.cuda.current_device()))
        print('Device count: ' + str(torch.cuda.device_count()))
        mlp.cuda()
        criterion = criterion.cuda()

        ### Shuffling experiment ###
        if do_shuffle:
            print("Current shuffling order {}".format(order))
            mlp.eval()  # prep model for *evaluation*
            test_loss = 0.0
            correct = 0
            total = 0
            for data, target in testloader:
                total += data.size(0)
                # use gpu
                data = data.cuda()
                target = target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output, _ = mlp(data)
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

            print('Testing Loss: {:.6f} \tTesting Accuracy: {:.6f}'.format(test_loss, test_acc))

            results[k]['Direct_eval'].append(test_acc)

        if continue_train:
            print("Continue training...")
            # specify optimizer
            optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

            for epoch in range(continue_train_epochs):

                mlp.train()  # prep model for training
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
                    output, _ = mlp(data)
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

                mlp.eval()  # prep model for *evaluation*
                test_loss = 0.0
                correct = 0
                total = 0
                for data, target in testloader:
                    total += data.size(0)
                    # use gpu
                    data = data.cuda()
                    target = target.cuda()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output, _ = mlp(data)
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

                if epoch + 1 == 1:
                    results[k]['Score_1_epoch'].append(test_acc)
                if epoch + 1 == 5:
                    results[k]['Score_5_epoch'].append(test_acc)

save_folder = 'shuffle_direct_eval/{}/'.format(model_name)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
joblib.dump(results, save_folder + 'shuffle_results.pickle')
