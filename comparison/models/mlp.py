import torch
import torch.nn as nn
import torch.nn.functional as F


# MNIST
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 768)
        self.layer2 = nn.Linear(768, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = self.layer1(x)
        extract_embedding.append(out)
        out = F.relu(out)
        extract_embedding.append(out)
        out = self.layer2(out)
        extract_embedding.append(out)
        out = F.softmax(out, dim=-1)
        return out, extract_embedding


class MLP_dropout_before_relu(nn.Module):
    def __init__(self, p):
        super(MLP_dropout_before_relu, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 768)
        self.dropout = nn.Dropout(p)
        self.layer2 = nn.Linear(768, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = self.layer1(x)
        extract_embedding.append(out)
        out = self.dropout(out)
        extract_embedding.append(out)
        out = F.relu(out)
        extract_embedding.append(out)
        out = self.layer2(out)
        extract_embedding.append(out)
        out = F.softmax(out, dim=-1)
        return out, extract_embedding


class MLP_dropout_after_relu(nn.Module):
    def __init__(self, p):
        super(MLP_dropout_after_relu, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 768)
        self.dropout = nn.Dropout(p)
        self.layer2 = nn.Linear(768, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = self.layer1(x)
        extract_embedding.append(out)
        out = F.relu(out)
        extract_embedding.append(out)
        out = self.dropout(out)
        extract_embedding.append(out)
        out = self.layer2(out)
        extract_embedding.append(out)
        out = F.softmax(out, dim=-1)
        return out, extract_embedding


class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 278)
        self.layer2 = nn.Linear(278, 278)
        self.layer3 = nn.Linear(278, 278)
        self.layer4 = nn.Linear(278, 278)
        self.layer5 = nn.Linear(278, 278)
        self.layer6 = nn.Linear(278, 278)
        self.layer7 = nn.Linear(278, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = F.relu(self.layer1(x))
        extract_embedding.append(out)
        out = F.relu(self.layer2(out))
        extract_embedding.append(out)
        out = F.relu(self.layer3(out))
        extract_embedding.append(out)
        out = F.relu(self.layer4(out))
        extract_embedding.append(out)
        out = F.relu(self.layer5(out))
        extract_embedding.append(out)
        out = F.relu(self.layer6(out))
        extract_embedding.append(out)
        out = self.layer7(out)
        out = F.softmax(out, dim=-1)
        extract_embedding.append(out)
        return out, extract_embedding


class DeepMLP_Long(nn.Module):
    def __init__(self):
        super(DeepMLP_Long, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 202)
        self.layer2 = nn.Linear(202, 202)
        self.layer3 = nn.Linear(202, 202)
        self.layer4 = nn.Linear(202, 202)
        self.layer5 = nn.Linear(202, 202)
        self.layer6 = nn.Linear(202, 202)
        self.layer7 = nn.Linear(202, 202)
        self.layer8 = nn.Linear(202, 202)
        self.layer9 = nn.Linear(202, 202)
        self.layer10 = nn.Linear(202, 202)
        self.layer11 = nn.Linear(202, 202)
        self.layer12 = nn.Linear(202, 202)
        self.layer13 = nn.Linear(202, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = F.relu(self.layer1(x))
        extract_embedding.append(out)
        out = F.relu(self.layer2(out))
        extract_embedding.append(out)
        out = F.relu(self.layer3(out))
        extract_embedding.append(out)
        out = F.relu(self.layer4(out))
        extract_embedding.append(out)
        out = F.relu(self.layer5(out))
        extract_embedding.append(out)
        out = F.relu(self.layer6(out))
        extract_embedding.append(out)
        out = F.relu(self.layer7(out))
        extract_embedding.append(out)
        out = F.relu(self.layer8(out))
        extract_embedding.append(out)
        out = F.relu(self.layer9(out))
        extract_embedding.append(out)
        out = F.relu(self.layer10(out))
        extract_embedding.append(out)
        out = F.relu(self.layer11(out))
        extract_embedding.append(out)
        out = F.relu(self.layer12(out))
        extract_embedding.append(out)
        out = self.layer13(out)
        out = F.softmax(out, dim=-1)
        extract_embedding.append(out)
        return out, extract_embedding


class DeepMLP_LN(nn.Module):
    def __init__(self):
        super(DeepMLP_LN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 278)
        self.layernorm1 = nn.LayerNorm(278)
        self.layer2 = nn.Linear(278, 278)
        self.layernorm2 = nn.LayerNorm(278)
        self.layer3 = nn.Linear(278, 278)
        self.layernorm3 = nn.LayerNorm(278)
        self.layer4 = nn.Linear(278, 278)
        self.layernorm4 = nn.LayerNorm(278)
        self.layer5 = nn.Linear(278, 278)
        self.layernorm5 = nn.LayerNorm(278)
        self.layer6 = nn.Linear(278, 278)
        self.layernorm6 = nn.LayerNorm(278)
        self.layer7 = nn.Linear(278, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = self.layernorm1(F.relu(self.layer1(x)))
        extract_embedding.append(out)
        out = self.layernorm2(F.relu(self.layer2(out)))
        extract_embedding.append(out)
        out = self.layernorm3(F.relu(self.layer3(out)))
        extract_embedding.append(out)
        out = self.layernorm4(F.relu(self.layer4(out)))
        extract_embedding.append(out)
        out = self.layernorm5(F.relu(self.layer5(out)))
        extract_embedding.append(out)
        out = self.layernorm6(F.relu(self.layer6(out)))
        extract_embedding.append(out)
        out = self.layer7(out)
        out = F.softmax(out, dim=-1)
        extract_embedding.append(out)
        return out, extract_embedding


class DeepMLP_Long_LN(nn.Module):
    def __init__(self):
        super(DeepMLP_Long_LN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 202)
        self.layernorm1 = nn.LayerNorm(202)
        self.layer2 = nn.Linear(202, 202)
        self.layernorm2 = nn.LayerNorm(202)
        self.layer3 = nn.Linear(202, 202)
        self.layernorm3 = nn.LayerNorm(202)
        self.layer4 = nn.Linear(202, 202)
        self.layernorm4 = nn.LayerNorm(202)
        self.layer5 = nn.Linear(202, 202)
        self.layernorm5 = nn.LayerNorm(202)
        self.layer6 = nn.Linear(202, 202)
        self.layernorm6 = nn.LayerNorm(202)
        self.layer7 = nn.Linear(202, 202)
        self.layernorm7 = nn.LayerNorm(202)
        self.layer8 = nn.Linear(202, 202)
        self.layernorm8 = nn.LayerNorm(202)
        self.layer9 = nn.Linear(202, 202)
        self.layernorm9 = nn.LayerNorm(202)
        self.layer10 = nn.Linear(202, 202)
        self.layernorm10 = nn.LayerNorm(202)
        self.layer11 = nn.Linear(202, 202)
        self.layernorm11 = nn.LayerNorm(202)
        self.layer12 = nn.Linear(202, 202)
        self.layernorm12 = nn.LayerNorm(202)
        self.layer13 = nn.Linear(202, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = self.layernorm1(F.relu(self.layer1(x)))
        extract_embedding.append(out)
        out = self.layernorm2(F.relu(self.layer2(out)))
        extract_embedding.append(out)
        out = self.layernorm3(F.relu(self.layer3(out)))
        extract_embedding.append(out)
        out = self.layernorm4(F.relu(self.layer4(out)))
        extract_embedding.append(out)
        out = self.layernorm5(F.relu(self.layer5(out)))
        extract_embedding.append(out)
        out = self.layernorm6(F.relu(self.layer6(out)))
        extract_embedding.append(out)
        out = self.layernorm7(F.relu(self.layer7(out)))
        extract_embedding.append(out)
        out = self.layernorm8(F.relu(self.layer8(out)))
        extract_embedding.append(out)
        out = self.layernorm9(F.relu(self.layer9(out)))
        extract_embedding.append(out)
        out = self.layernorm10(F.relu(self.layer10(out)))
        extract_embedding.append(out)
        out = self.layernorm11(F.relu(self.layer11(out)))
        extract_embedding.append(out)
        out = self.layernorm12(F.relu(self.layer12(out)))
        extract_embedding.append(out)
        out = self.layer13(out)
        out = F.softmax(out, dim=-1)
        extract_embedding.append(out)
        return out, extract_embedding


class DeepMLP_LN_SC(nn.Module):
    def __init__(self):
        super(DeepMLP_LN_SC, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 278)
        self.layernorm1 = nn.LayerNorm(278)
        self.layer2 = nn.Linear(278, 278)
        self.layernorm2 = nn.LayerNorm(278)
        self.layer3 = nn.Linear(278, 278)
        self.layernorm3 = nn.LayerNorm(278)
        self.layer4 = nn.Linear(278, 278)
        self.layernorm4 = nn.LayerNorm(278)
        self.layer5 = nn.Linear(278, 278)
        self.layernorm5 = nn.LayerNorm(278)
        self.layer6 = nn.Linear(278, 278)
        self.layernorm6 = nn.LayerNorm(278)
        self.layer7 = nn.Linear(278, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = self.layernorm1(F.relu(self.layer1(x)))
        extract_embedding.append(out)
        out = self.layernorm2(F.relu(self.layer2(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm3(F.relu(self.layer3(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm4(F.relu(self.layer4(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm5(F.relu(self.layer5(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm6(F.relu(self.layer6(out)) + out)
        extract_embedding.append(out)
        out = self.layer7(out)
        out = F.softmax(out, dim=-1)
        extract_embedding.append(out)
        return out, extract_embedding


class DeepMLP_Long_LN_SC(nn.Module):
    def __init__(self):
        super(DeepMLP_Long_LN_SC, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 202)
        self.layernorm1 = nn.LayerNorm(202)
        self.layer2 = nn.Linear(202, 202)
        self.layernorm2 = nn.LayerNorm(202)
        self.layer3 = nn.Linear(202, 202)
        self.layernorm3 = nn.LayerNorm(202)
        self.layer4 = nn.Linear(202, 202)
        self.layernorm4 = nn.LayerNorm(202)
        self.layer5 = nn.Linear(202, 202)
        self.layernorm5 = nn.LayerNorm(202)
        self.layer6 = nn.Linear(202, 202)
        self.layernorm6 = nn.LayerNorm(202)
        self.layer7 = nn.Linear(202, 202)
        self.layernorm7 = nn.LayerNorm(202)
        self.layer8 = nn.Linear(202, 202)
        self.layernorm8 = nn.LayerNorm(202)
        self.layer9 = nn.Linear(202, 202)
        self.layernorm9 = nn.LayerNorm(202)
        self.layer10 = nn.Linear(202, 202)
        self.layernorm10 = nn.LayerNorm(202)
        self.layer11 = nn.Linear(202, 202)
        self.layernorm11 = nn.LayerNorm(202)
        self.layer12 = nn.Linear(202, 202)
        self.layernorm12 = nn.LayerNorm(202)
        self.layer13 = nn.Linear(202, 10)

    def forward(self, x):
        extract_embedding = []
        x = x.view(-1, 28 * 28)
        extract_embedding.append(x)
        out = self.layernorm1(F.relu(self.layer1(x)))
        extract_embedding.append(out)
        out = self.layernorm2(F.relu(self.layer2(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm3(F.relu(self.layer3(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm4(F.relu(self.layer4(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm5(F.relu(self.layer5(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm6(F.relu(self.layer6(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm7(F.relu(self.layer7(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm8(F.relu(self.layer8(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm9(F.relu(self.layer9(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm10(F.relu(self.layer10(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm11(F.relu(self.layer11(out)) + out)
        extract_embedding.append(out)
        out = self.layernorm12(F.relu(self.layer12(out)) + out)
        extract_embedding.append(out)
        out = self.layer13(out)
        out = F.softmax(out, dim=-1)
        extract_embedding.append(out)
        return out, extract_embedding


if __name__ == "__main__":
    net = DeepMLP_LN()
    print(net.__class__.__name__)
    # x = torch.ones((2, 1, 28, 28))
    # y, layer_embeddings = net(x)
    # print(y.shape)
    # print([t.shape for t in layer_embeddings])
    # for name, param in net.named_parameters():
    #     print(name)

