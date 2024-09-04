'''

    This script is used to perturb and amplify the data, and label the data to
    facilitate the prediction of perturb data in the next step.

'''


import csv
import os
import random
from dataclasses import dataclass, field
import time
import transformers
from tqdm import tqdm


def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return hours, minutes, seconds


@dataclass
class DataArguments:
    data_path: str = field(default=r"..\data", metadata={"help": "Path to the training data."})


def get_pos_data(file_path):
    sub_dataset = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # skip header
        for columns in tqdm(reader, desc="Data read progress"):
            sequence = columns[0]
            label = int(columns[1])
            if label == 1:
                sub_dataset.append([sequence, label])
    return sub_dataset


def work():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()
    if isinstance(data_args, tuple):
        data_args = data_args[0]

    train_data = get_pos_data(os.path.join(data_args.data_path, "train.csv"))
    dev_data = get_pos_data(os.path.join(data_args.data_path, "dev.csv"))
    test_data = get_pos_data(os.path.join(data_args.data_path, "test.csv"))

    dataset = train_data + dev_data + test_data

    mapping_array = ['A', 'C', 'G', 'T']
    out_file = os.path.join(data_args.data_path, "dataset_PA.csv")
    post_perturbation_data = []

    for row in tqdm(dataset, desc="Data PA progress"):
        sequence = row[0]
        label = row[1]
        post_perturbation_data.append([sequence, label, 0])  # Preserving the original data
        length = len(sequence)
        center_idx = length // 2  # total 201bp center is 101, it starts at 0 so idx is 100
        hexamer_start = 0
        hexamer_end = hexamer_start + 6
        while hexamer_end <= length:
            temp = sequence
            original_hexamer_base = temp[hexamer_start:hexamer_end]

            random_num = 100
            for i in range(0, random_num):
                random_hexamer_base = [mapping_array[random.randint(0, 3)] for _ in range(6)]  # 6 random bases
                temp = list(temp)
                temp[hexamer_start:hexamer_end] = random_hexamer_base
                temp = ''.join(temp)
                relative_pos = hexamer_start - center_idx
                # Data format: [sequence, label, original or perturbed label (0/1), original hexamer, relative position]
                post_perturbation_data.append([temp, label, 1, original_hexamer_base, relative_pos])

            hexamer_start += 1
            hexamer_end += 1

    print(f'Number of positive samples in the dataset after perturbation:{len(post_perturbation_data)}')

    with open(out_file, 'w', newline='') as outfile:
        target_writer = csv.writer(outfile, delimiter=',')
        target_writer.writerow(['sequence', 'label', 'perturbation_flag', 'original_hexamer_base', 'relative_pos', 'pre_total:' + str(len(dataset)), 'post_total:' + str(len(post_perturbation_data))])

        for row in tqdm(post_perturbation_data, desc="Data writing progress"):
            target_writer.writerow(row)

    print(f"finished")


if __name__ == "__main__":
    start_time = time.time()

    work()

    end_time = time.time()
    run_time = end_time - start_time  # Calculate the running time in seconds
    print("program start time:", time.ctime(start_time))
    print("program end time:", time.ctime(end_time))
    hours, minutes, seconds = convert_seconds(run_time)
    print(f"Run time: {hours}hours {minutes}minutes {seconds}seconds")


