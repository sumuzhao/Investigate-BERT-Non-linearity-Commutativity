import torch
import torch.nn as nn
import torch.nn.functional as F


# MNIST
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        extract_embedding = []
        x = F.relu(self.conv1(x))
        extract_embedding.append(x)
        x = F.max_pool2d(x, (2, 2))
        extract_embedding.append(x)
        x = F.relu(self.conv2(x))
        extract_embedding.append(x)
        x = F.max_pool2d(x, (2, 2))
        extract_embedding.append(x)
        x = x.view(-1, 16 * 5 * 5)
        extract_embedding.append(x)
        x = F.relu(self.fc1(x))
        extract_embedding.append(x)
        x = F.relu(self.fc2(x))
        extract_embedding.append(x)
        x = self.fc3(x)
        extract_embedding.append(x)
        return x


# MNIST or CIFAR10
class ConvNet(nn.Module):
    def __init__(self, c=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=8, kernel_size=9, stride=2, padding=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(1152, 10)

    def forward(self, x):
        extract_embedding = []
        x = F.relu(self.conv1(x))
        extract_embedding.append(x)
        x = F.relu(self.conv2(x))
        extract_embedding.append(x)
        x = F.relu(self.conv3(x))
        extract_embedding.append(x)
        x = F.relu(self.conv4(x))
        extract_embedding.append(x)
        x = F.relu(self.conv5(x))
        extract_embedding.append(x)
        x = F.relu(self.conv6(x))
        extract_embedding.append(x)
        x = x.view(-1, 8 * 12 * 12)
        extract_embedding.append(x)
        x = self.fc1(x)
        extract_embedding.append(x)
        return x


class ConvNet_Long(nn.Module):
    def __init__(self, c=1):
        super(ConvNet_Long, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=8, kernel_size=9, stride=2, padding=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv9 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv11 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(1152, 10)

    def forward(self, x):
        extract_embedding = []
        x = F.relu(self.conv1(x))
        extract_embedding.append(x)
        x = F.relu(self.conv2(x))
        extract_embedding.append(x)
        x = F.relu(self.conv3(x))
        extract_embedding.append(x)
        x = F.relu(self.conv4(x))
        extract_embedding.append(x)
        x = F.relu(self.conv5(x))
        extract_embedding.append(x)
        x = F.relu(self.conv6(x))
        extract_embedding.append(x)
        x = F.relu(self.conv7(x))
        extract_embedding.append(x)
        x = F.relu(self.conv8(x))
        extract_embedding.append(x)
        x = F.relu(self.conv9(x))
        extract_embedding.append(x)
        x = F.relu(self.conv10(x))
        extract_embedding.append(x)
        x = F.relu(self.conv11(x))
        extract_embedding.append(x)
        x = F.relu(self.conv12(x))
        extract_embedding.append(x)
        x = x.view(-1, 8 * 12 * 12)
        extract_embedding.append(x)
        x = self.fc1(x)
        extract_embedding.append(x)
        return x


class ConvNet_LN(nn.Module):
    def __init__(self, c=1):
        super(ConvNet_LN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=8, kernel_size=9, stride=2, padding=2, bias=True)
        self.layernorm1 = nn.LayerNorm([8, 12, 12])
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm2 = nn.LayerNorm([8, 12, 12])
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm3 = nn.LayerNorm([8, 12, 12])
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm4 = nn.LayerNorm([8, 12, 12])
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm5 = nn.LayerNorm([8, 12, 12])
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm6 = nn.LayerNorm([8, 12, 12])
        self.fc1 = nn.Linear(1152, 10)

    def forward(self, x):
        extract_embedding = []
        x = self.layernorm1(F.relu(self.conv1(x)))
        extract_embedding.append(x)
        x = self.layernorm2(F.relu(self.conv2(x)))
        extract_embedding.append(x)
        x = self.layernorm3(F.relu(self.conv3(x)))
        extract_embedding.append(x)
        x = self.layernorm4(F.relu(self.conv4(x)))
        extract_embedding.append(x)
        x = self.layernorm5(F.relu(self.conv5(x)))
        extract_embedding.append(x)
        x = self.layernorm6(F.relu(self.conv6(x)))
        extract_embedding.append(x)
        x = x.view(-1, 8 * 12 * 12)
        extract_embedding.append(x)
        x = self.fc1(x)
        extract_embedding.append(x)
        return x


class ConvNet_Long_LN(nn.Module):
    def __init__(self, c=1):
        super(ConvNet_Long_LN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=8, kernel_size=9, stride=2, padding=2, bias=True)
        self.layernorm1 = nn.LayerNorm([8, 12, 12])
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm2 = nn.LayerNorm([8, 12, 12])
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm3 = nn.LayerNorm([8, 12, 12])
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm4 = nn.LayerNorm([8, 12, 12])
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm5 = nn.LayerNorm([8, 12, 12])
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm6 = nn.LayerNorm([8, 12, 12])
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm7 = nn.LayerNorm([8, 12, 12])
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm8 = nn.LayerNorm([8, 12, 12])
        self.conv9 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm9 = nn.LayerNorm([8, 12, 12])
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm10 = nn.LayerNorm([8, 12, 12])
        self.conv11 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm11 = nn.LayerNorm([8, 12, 12])
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm12 = nn.LayerNorm([8, 12, 12])
        self.fc1 = nn.Linear(1152, 10)

    def forward(self, x):
        extract_embedding = []
        x = self.layernorm1(F.relu(self.conv1(x)))
        extract_embedding.append(x)
        x = self.layernorm2(F.relu(self.conv2(x)))
        extract_embedding.append(x)
        x = self.layernorm3(F.relu(self.conv3(x)))
        extract_embedding.append(x)
        x = self.layernorm4(F.relu(self.conv4(x)))
        extract_embedding.append(x)
        x = self.layernorm5(F.relu(self.conv5(x)))
        extract_embedding.append(x)
        x = self.layernorm6(F.relu(self.conv6(x)))
        extract_embedding.append(x)
        x = self.layernorm7(F.relu(self.conv7(x)))
        extract_embedding.append(x)
        x = self.layernorm8(F.relu(self.conv8(x)))
        extract_embedding.append(x)
        x = self.layernorm9(F.relu(self.conv9(x)))
        extract_embedding.append(x)
        x = self.layernorm10(F.relu(self.conv10(x)))
        extract_embedding.append(x)
        x = self.layernorm11(F.relu(self.conv11(x)))
        extract_embedding.append(x)
        x = self.layernorm12(F.relu(self.conv12(x)))
        extract_embedding.append(x)
        x = x.view(-1, 8 * 12 * 12)
        extract_embedding.append(x)
        x = self.fc1(x)
        extract_embedding.append(x)
        return x


class ConvNet_LN_SC(nn.Module):
    def __init__(self, c=1):
        super(ConvNet_LN_SC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=8, kernel_size=9, stride=2, padding=2, bias=True)
        self.layernorm1 = nn.LayerNorm([8, 12, 12])
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm2 = nn.LayerNorm([8, 12, 12])
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm3 = nn.LayerNorm([8, 12, 12])
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm4 = nn.LayerNorm([8, 12, 12])
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm5 = nn.LayerNorm([8, 12, 12])
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm6 = nn.LayerNorm([8, 12, 12])
        self.fc1 = nn.Linear(1152, 10)

    def forward(self, x):
        extract_embedding = []
        x = self.layernorm1(F.relu(self.conv1(x)))
        extract_embedding.append(x)
        x = self.layernorm2(F.relu(self.conv2(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm3(F.relu(self.conv3(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm4(F.relu(self.conv4(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm5(F.relu(self.conv5(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm6(F.relu(self.conv6(x)) + x)
        extract_embedding.append(x)
        x = x.view(-1, 8 * 12 * 12)
        extract_embedding.append(x)
        x = self.fc1(x)
        extract_embedding.append(x)
        return x


class ConvNet_Long_LN_SC(nn.Module):
    def __init__(self, c=1):
        super(ConvNet_Long_LN_SC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=8, kernel_size=9, stride=2, padding=2, bias=True)
        self.layernorm1 = nn.LayerNorm([8, 12, 12])
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm2 = nn.LayerNorm([8, 12, 12])
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm3 = nn.LayerNorm([8, 12, 12])
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm4 = nn.LayerNorm([8, 12, 12])
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm5 = nn.LayerNorm([8, 12, 12])
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm6 = nn.LayerNorm([8, 12, 12])
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm7 = nn.LayerNorm([8, 12, 12])
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm8 = nn.LayerNorm([8, 12, 12])
        self.conv9 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm9 = nn.LayerNorm([8, 12, 12])
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm10 = nn.LayerNorm([8, 12, 12])
        self.conv11 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm11 = nn.LayerNorm([8, 12, 12])
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm12 = nn.LayerNorm([8, 12, 12])
        self.fc1 = nn.Linear(1152, 10)

    def forward(self, x):
        extract_embedding = []
        x = self.layernorm1(F.relu(self.conv1(x)))
        extract_embedding.append(x)
        x = self.layernorm2(F.relu(self.conv2(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm3(F.relu(self.conv3(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm4(F.relu(self.conv4(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm5(F.relu(self.conv5(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm6(F.relu(self.conv6(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm7(F.relu(self.conv7(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm8(F.relu(self.conv8(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm9(F.relu(self.conv9(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm10(F.relu(self.conv10(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm11(F.relu(self.conv11(x)) + x)
        extract_embedding.append(x)
        x = self.layernorm12(F.relu(self.conv12(x)) + x)
        extract_embedding.append(x)
        x = x.view(-1, 8 * 12 * 12)
        extract_embedding.append(x)
        x = self.fc1(x)
        extract_embedding.append(x)
        return x


class ConvNet_LN_Residual(nn.Module):
    def __init__(self, c=1):
        super(ConvNet_LN_Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=8, kernel_size=9, stride=2, padding=2, bias=True)
        self.layernorm1 = nn.LayerNorm([8, 12, 12])
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm2 = nn.LayerNorm([8, 12, 12])
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm3 = nn.LayerNorm([8, 12, 12])
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm4 = nn.LayerNorm([8, 12, 12])
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm5 = nn.LayerNorm([8, 12, 12])
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.layernorm6 = nn.LayerNorm([8, 12, 12])
        self.fc1 = nn.Linear(1152, 10)

    def forward(self, x):
        extract_embedding = []
        x1 = self.layernorm1(F.relu(self.conv1(x)))
        extract_embedding.append(x1)
        x2 = self.layernorm2(F.relu(self.conv2(x1)))
        extract_embedding.append(x2)
        x3 = self.layernorm3(F.relu(self.conv3(x2)) + x1)
        extract_embedding.append(x3)
        x4 = self.layernorm4(F.relu(self.conv4(x3)))
        extract_embedding.append(x4)
        x5 = self.layernorm5(F.relu(self.conv5(x4)))
        extract_embedding.append(x5)
        x6 = self.layernorm6(F.relu(self.conv6(x5)) + x4)
        extract_embedding.append(x6)
        x6 = x6.view(-1, 8 * 12 * 12)
        extract_embedding.append(x6)
        x7 = self.fc1(x6)
        extract_embedding.append(x7)
        return x7


if __name__ == "__main__":
    net = ConvNet_LN()
    print(net)
    x = torch.ones((2, 1, 28, 28))
    y, layer_embeddings = net(x)
    print(y.shape)
    print([t.shape for t in layer_embeddings])
    # for name, param in net.named_parameters():
    #     print(name)
