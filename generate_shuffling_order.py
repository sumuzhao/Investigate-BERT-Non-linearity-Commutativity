import numpy as np
from itertools import combinations


# for BERT-base
# layer_orders = {k: set() for k in range(10, -1, -1)}
# layers = list(range(12))
# times = 100
# np.random.seed(0)
#
# layer_orders[10] = []
# for combination in combinations(layers, 10):
#     fixed_layers = list(combination)
#     rest_layers = [l for l in layers if l not in fixed_layers]
#     order = list(range(12))
#     order[rest_layers[0]], order[rest_layers[1]] = order[rest_layers[1]], order[rest_layers[0]]
#     layer_orders[10].append(','.join(map(str, order)))
# supplement_idx = np.random.choice(len(layer_orders[10]), times - len(layer_orders[10]), replace=False)
# layer_orders[10].extend([layer_orders[10][idx] for idx in supplement_idx])
# print(len(layer_orders[10]))
#
# for k in range(9, -1, -1):
#     while len(layer_orders[k]) < 100:
#         fixed_layers = sorted(np.random.choice(12, k, replace=False))
#         rest_layers = [l for l in layers if l not in fixed_layers]
#         np.random.shuffle(rest_layers)
#         order = list(range(12))
#         rest_idx = 0
#         for i, l in enumerate(layers):
#             if l not in fixed_layers and l in rest_layers:
#                 order[i] = rest_layers[rest_idx]
#                 rest_idx += 1
#         layer_orders[k].add(','.join(map(str, order)))
#
# for k in range(9, -1, -1):
#     layer_orders[k] = list(layer_orders[k])
#
# print(layer_orders)
# print([len(layer_orders[k]) for k in range(10, -1, -1)])
#
# for k in range(10, -1, -1):
#     with open('layer-commutativity/shuffle_exhaustive_search/fix{}_layers.txt'.format(k), 'w') as f:
#         f.writelines('\n'.join(layer_orders[k]))


# for 12-layer CNN
layer_orders = {k: set() for k in range(9, -1, -1)}
layers = list(range(1, 12))
times = 100
np.random.seed(0)

layer_orders[9] = []
for combination in combinations(layers, 9):
    fixed_layers = list(combination)
    rest_layers = [l for l in layers if l not in fixed_layers]
    order = list(range(1, 12))
    order[rest_layers[0] - 1], order[rest_layers[1] - 1] = order[rest_layers[1] - 1], order[rest_layers[0] - 1]
    order = [e + 1 for e in order]
    layer_orders[9].append('1,' + ','.join(map(str, order)))
supplement_idx = np.random.choice(len(layer_orders[9]), times - len(layer_orders[9]), replace=False)
layer_orders[9].extend([layer_orders[9][idx] for idx in supplement_idx])
print(len(layer_orders[9]))

for k in range(8, -1, -1):
    while len(layer_orders[k]) < 100:
        fixed_layers = sorted(np.random.choice(range(1, 12), k, replace=False))
        rest_layers = [l for l in layers if l not in fixed_layers]
        np.random.shuffle(rest_layers)
        order = list(range(1, 12))
        rest_idx = 0
        for i, l in enumerate(layers):
            if l not in fixed_layers and l in rest_layers:
                order[i] = rest_layers[rest_idx]
                rest_idx += 1
        order = [e + 1 for e in order]
        layer_orders[k].add('1,' + ','.join(map(str, order)))

for k in range(9, -1, -1):
    layer_orders[k] = list(layer_orders[k])

print(layer_orders[0][:5])
print([len(layer_orders[k]) for k in range(9, -1, -1)])
#
for k in range(9, -1, -1):
    with open('comparison/shuffle_exhaustive_search/fix{}_layers.txt'.format(k), 'w') as f:
        f.writelines('\n'.join(layer_orders[k]))
