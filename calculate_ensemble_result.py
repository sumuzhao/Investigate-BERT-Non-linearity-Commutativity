"""
Calculate the metrics based on ensembling predictions of models with multiple random rums.
"""

import os
import argparse
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dir', dest='dir', default='', help='path of the target directory')
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.dir):
        print("Dir error. ")
    else:

        # prediction_file = []
        # for root, dirs, _ in os.walk(args.dir):
        #     if root == args.dir:
        #         for dir in dirs:
        #             prediction_file.append(os.path.join(root, dir, "dev_predictions.tsv"))
        # print(prediction_file)
        #
        # with open(prediction_file[0], "r") as reader:
        #     content = reader.readlines()
        #     num_examples = len(content) - 1
        #     preds = np.zeros([num_examples, len(prediction_file)])
        #     label = np.zeros(num_examples)
        #
        #     for i, line in enumerate(content):
        #         if i == 0: continue
        #         parts = line.split('\t')
        #         preds[i - 1][0] = int(parts[1])
        #         label[i - 1] = int(parts[3])
        #
        # for j, file in enumerate(prediction_file):
        #     if j == 0: continue
        #     with open(file, "r") as reader:
        #         for i, line in enumerate(reader.readlines()):
        #             if i == 0: continue
        #             parts = line.split('\t')
        #             preds[i - 1][j] = int(parts[1])
        #
        # pred = np.zeros_like(label)
        # for i in range(num_examples):
        #     count = Counter(preds[i]).most_common()
        #     if len(count) == 1:
        #         pred[i] = int(count[0][0])
        #     else:
        #         # currently only work on MNLI, which has three classes.
        #         # Each experiments has 10 runs,
        #         # so we don't have situations where three classes have the same voting numbers.
        #         if count[0][1] > count[1][1]:
        #             pred[i] = int(count[0][0])
        #         else:
        #             random_val = np.random.randint(0, 1)
        #             pred[i] = int(count[0][0]) if random_val >= 0.5 else int(count[1][0])
        #
        #     # if pred[i] != label[i]:
        #     #     print(preds[i], count, "pred", pred[i], "label", label[i])
        #
        # accuracy = round(np.sum(np.equal(pred, label)) / num_examples, 4)
        # print("The ensemble accuracy is {}".format(accuracy))
        # f1 = round(f1_score(pred, label), 4)
        # print("The ensemble f1 score is {}".format(f1))

        prediction_files = [[] for _ in range(12)]
        for root, dirs, _ in os.walk(args.dir):
            if root == args.dir:
                for dir in dirs:
                    layer = int(dir.split('_')[1])
                    prediction_files[layer].append(os.path.join(root, dir, "dev_predictions.tsv"))
        # print(prediction_files)

        for layer, prediction_file in enumerate(prediction_files):
            with open(prediction_file[0], "r") as reader:
                content = reader.readlines()
                num_examples = len(content) - 1
                preds = np.zeros([num_examples, len(prediction_file)])
                label = np.zeros(num_examples)

                for i, line in enumerate(content):
                    if i == 0: continue
                    parts = line.split('\t')
                    preds[i - 1][0] = int(parts[1])
                    label[i - 1] = int(parts[3])

            for j, file in enumerate(prediction_file):
                if j == 0: continue
                with open(file, "r") as reader:
                    for i, line in enumerate(reader.readlines()):
                        if i == 0: continue
                        parts = line.split('\t')
                        preds[i - 1][j] = int(parts[1])

            pred = np.zeros_like(label)
            for i in range(num_examples):
                count = Counter(preds[i]).most_common()
                if len(count) == 1:
                    pred[i] = int(count[0][0])
                else:
                    # currently only work on MNLI, which has three classes.
                    # Each experiments has 10 runs,
                    # so we don't have situations where three classes have the same voting numbers.
                    if count[0][1] > count[1][1]:
                        pred[i] = int(count[0][0])
                    else:
                        random_val = np.random.randint(0, 1)
                        pred[i] = int(count[0][0]) if random_val >= 0.5 else int(count[1][0])

                # if pred[i] != label[i]:
                #     print(preds[i], count, "pred", pred[i], "label", label[i])

            accuracy = round(np.sum(np.equal(pred, label)) / num_examples, 4)
            print("The ensemble accuracy for {} is {}".format(layer, accuracy))
            # f1 = round(f1_score(pred, label), 4)
            # print("The ensemble f1 score is {}".format(f1))
