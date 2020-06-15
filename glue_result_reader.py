"""Reader the results generated by fine-tuning"""

import os
import shutil
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dir', dest='dir', default='', help='path of the fine-tuning results')
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.dir):
        print("Dir error. ")
    else:

        # task_name --> standard task name
        standard_file_name = {
            "cola": "CoLA",
            "mnlim": "MNLI-m",
            "mnlimm": "MNLI-mm",
            "mrpc": "MRPC",
            "qnli": "QNLI",
            "qqp": "QQP",
            "rte": "RTE",
            "sst2": "SST-2",
            "stsb": "STS-B",
            "wnli": "WNLI",
            "ax": "AX"}


        task_folders = []
        root_dir = ""
        for root, dirs, _ in os.walk(args.dir):
            if root == args.dir:
                root_dir = root
                task_folders = dirs
                break
        task_folders = sorted([task_folder for task_folder in task_folders if task_folder != "submission"])
        print("Root dir:", root_dir)
        print("Task folders: ", task_folders)

        for task_folder in task_folders:

            task_name = task_folder.split('_')[0]
            print("Current task: {}".format(task_name))

            try:
                if task_name != 'ax':
                    with open(os.path.join(root_dir, task_folder) + "/eval_results.txt", "r") as reader:
                        for line in reader.readlines():
                            print(line)

                shutil.copyfile(os.path.join(root_dir, task_folder) + "/{}.tsv".format(standard_file_name[task_name]),
                                os.path.join(root_dir, "submission") + "/{}.tsv".format(standard_file_name[task_name]))
            except:
                continue

