'''

    This script is used to perform parallel prediction on the perturbed data,
    here using the dnabert2 model, while storing the prediction results.

'''
import csv
import math
import random
from datetime import datetime
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
from tqdm import tqdm
import transformers
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=r"F:\YDDataSet\binary_class_best_model\checkpoint-3150")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=r"F:\YDDataSet\binary_class_best_model\10000", metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    num_labels: int = field(default=2)
    tag: int = field(default=1)
    exp_cnt_per_class: int = field(default=200)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="DNABERT2_run")  # The name of the folder to output the file
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=50, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=128)
    num_train_epochs: int = field(default=5)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)  # 100
    eval_steps: int = field(default=100)  # 100
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output/dnabert2")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    overwrite_output_dir: bool = field(default=True)
    log_level: str = field(default="info")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 texts: [str],
                 labels: [int],
                 num_labels: int,
                 tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedDataset, self).__init__()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = num_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def calculate_log_odds(probability):
    if probability == 1:
        return float('inf')
    elif probability == 0:
        return float('-inf')
    else:
        return math.log(probability / (1 - probability))


# Compute softmax results for logits
def softmax(logits):
    exp_logits = np.exp(logits)
    exp_sum = np.sum(exp_logits, axis=-1, keepdims=True)
    softmax_probs = exp_logits / exp_sum
    return softmax_probs


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    # if logits.ndim == 3:
    #     # Reshape logits to 2D if needed
    #     logits = logits.reshape(-1, logits.shape[-1])
    # predictions = np.argmax(logits, axis=-1)
    # valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    # valid_predictions = predictions[valid_mask]
    # valid_labels = labels[valid_mask]
    return {
        # "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        # "f1": sklearn.metrics.f1_score(
        #     valid_labels, valid_predictions, average="macro", zero_division=0
        # ),
        # "matthews_correlation": sklearn.metrics.matthews_corrcoef(
        #     valid_labels, valid_predictions
        # ),
        # "precision": sklearn.metrics.precision_score(
        #     valid_labels, valid_predictions, average="macro", zero_division=0
        # ),
        # "recall": sklearn.metrics.recall_score(
        #     valid_labels, valid_predictions, average="macro", zero_division=0
        # ),
    }


"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):  # Only required if using model evaluation
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def get_predict_correct_sample(trainer, data, labels):
    test_dataset = SupervisedDataset(tokenizer=trainer.tokenizer, texts=data, labels=labels,
                                     num_labels=trainer.model.num_labels)
    logits = trainer.predict(test_dataset).predictions
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    # Find the index with the correct prediction
    correct_predictions = [i for i in range(len(predictions)) if predictions[i] == labels[i]]
    # New datasets and labels are built using the filtered indices
    new_dataset = [data[i] for i in correct_predictions]  # new_dataset is the dataset for which the predictor predicted correctly
    new_labels = [labels[i] for i in correct_predictions]
    return new_dataset, new_labels


def get_prediction_log_odds(trainer, data, labels=None):
    test_dataset = SupervisedDataset(tokenizer=trainer.tokenizer, texts=data, labels=labels,
                                     num_labels=trainer.model.num_labels)
    logits = trainer.predict(test_dataset).predictions
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    logits = softmax(logits)

    # Get the log-odds of the sample with the correct prediction.
    log_odds = []
    extreme_val_cnt = 0
    for item in logits:
        correct_probability = item[1]  # The probability of getting a positive sample
        if correct_probability == 1.0 or correct_probability == 0.0:
            extreme_val_cnt += 1
        current_log_odds = calculate_log_odds(correct_probability)
        log_odds.append(current_log_odds)

    return log_odds, extreme_val_cnt


# Load the dataset - positive samples
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


def get_motif_loci_importance(cur_log_odds, original_motif_log_odds, perturbation_number, motif_loci):
    if (len(cur_log_odds) // perturbation_number) * perturbation_number != len(cur_log_odds):
        print(f'The total number of predicted results after perturbation is wrong!')

    motif_loci_importance = []
    # Changed log-odds array
    cur_log_odds = np.array(cur_log_odds)
    # Calculate the log-odds change
    delta_log_odds = cur_log_odds - original_motif_log_odds
    idx = 0
    start_idx = 0
    end_idx = perturbation_number
    length = len(motif_loci)
    while idx < length:
        median_delta_log_odds = np.median(delta_log_odds[start_idx:end_idx])
        # Stored in order of [motif, position, importance]
        motif_loci_importance.append([motif_loci[idx][0], motif_loci[idx][1], median_delta_log_odds])
        idx += 1
        start_idx += perturbation_number
        end_idx += perturbation_number

    return motif_loci_importance


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                           cache_dir=training_args.cache_dir,
                                                           model_max_length=training_args.model_max_length,
                                                           padding_side="right",
                                                           use_fast=True,
                                                           trust_remote_code=True,
                                                           )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=data_args.num_labels,
        trust_remote_code=True,
    )

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   data_collator=data_collator)

    # Checking the parallel number
    if data_args.tag < 1:
        print("tag is an integer greater than 0.")
        exit(1)

    # All positive samples are obtained
    train_data = get_pos_data(os.path.join(data_args.data_path, "train.csv"))
    dev_data = get_pos_data(os.path.join(data_args.data_path, "dev.csv"))
    test_data = get_pos_data(os.path.join(data_args.data_path, "test.csv"))

    dataset = train_data + dev_data + test_data
    data = [row[0] for row in dataset]
    labels = [row[1] for row in dataset]

    # Data extraction is performed according to the parallel number
    exp_cnt_per_class = data_args.exp_cnt_per_class  # Number of samples to be processed for each parallel class
    tag = data_args.tag
    start_idx = (tag - 1) * exp_cnt_per_class
    end_idx = start_idx + exp_cnt_per_class
    print(f'Perturbation enhancement and prediction from {start_idx + 1} to {end_idx} positive samples')
    current_data = data[start_idx:end_idx]
    current_labels = labels[start_idx:end_idx]

    # Retain all samples with correct predictions
    original_data_size = len(current_data)
    current_data, current_labels = get_predict_correct_sample(trainer, current_data, current_labels)
    correct_predict_size = len(current_data)

    accuracy = correct_predict_size / original_data_size
    formatted_accuracy = "{:.2%}".format(accuracy)

    print(f'Total: {original_data_size} samples')
    print(f'Number of correct predictions: {correct_predict_size} samples')
    print(f'Correct prediction rate: {formatted_accuracy}')

    raw_log_odds, raw_extreme_val_cnt = get_prediction_log_odds(trainer, current_data, current_labels)
    print(f'Number of -inf/inf extreme values in raw_log_odds: {raw_extreme_val_cnt}')

    extreme_val_cnt_total = 0
    mapping_array = ['A', 'C', 'G', 'T']

    post_perturbation_data = []  # The perturbed samples are stored

    random_num = 100   # Define the number of times each position-specific motif needs to be perturbed
    cur_perturb_labels = []
    if correct_predict_size > 0:
        max_size = len(current_data[0]) * random_num
        # The label of the perturbed sample of the current sample is stored, merely to keep the input consistent
        cur_perturb_labels = [1 for _ in range(max_size)]

    for i in tqdm(range(correct_predict_size), desc="Data perturbation progress"):
        sequence = current_data[i]
        arr_seq = list(sequence)
        original_motif_log_odds = raw_log_odds[i]
        motif_loci = []  # Store motif and position information for the current sample
        cur_perturb_data = []  # Store perturbed samples of the current sample

        length = len(sequence)
        center_idx = length // 2  # total 201bp, center is 101. It starts at 0 so idx is 100.
        hexamer_start = 0
        hexamer_end = hexamer_start + 6  # Left closed right open interval, hexamer.
        while hexamer_end <= length:
            original_hexamer_base = sequence[hexamer_start:hexamer_end]
            relative_pos = hexamer_start - center_idx
            # Storage format: [hexamers, relative position]
            motif_loci.append([original_hexamer_base, relative_pos])

            pre_string = ''.join(arr_seq[:hexamer_start])
            end_string = ''.join(arr_seq[hexamer_end:])

            for _ in range(random_num):
                random_hexamer_base = random.choices(mapping_array, k=6)
                sub_seq = pre_string + ''.join(random_hexamer_base) + end_string
                cur_perturb_data.append(sub_seq)

            hexamer_start += 1
            hexamer_end += 1

        # Predict the obtained perturbed data of all loci in the current sample and obtain its log_odds value
        cur_log_odds, cur_extreme_val_cnt = get_prediction_log_odds(trainer, cur_perturb_data, cur_perturb_labels)
        extreme_val_cnt_total += cur_extreme_val_cnt

        # For any position-specific motif, the difference between the prediction result after 100 perturbations and
        # the original sample prediction result is taken to obtain 100 change values, and then the median of
        # the change values is taken as the importance value of the position-specific motif.

        # Storage format: [hexamer, relative position, importance score]
        motif_loci_importance = get_motif_loci_importance(cur_log_odds, original_motif_log_odds, random_num, motif_loci)
        post_perturbation_data.append([sequence, motif_loci_importance])

    print(f'The number of -inf/inf extreme values in the predicted log_odds of all perturbed data: {extreme_val_cnt_total}')

    # Output prediction result
    out_file = os.path.join(training_args.output_dir, 'dataset_PA_post_t' + str(original_data_size) + '_o' + str(tag) + '.csv')
    with open(out_file, 'w', newline='') as outfile:
        target_writer = csv.writer(outfile, delimiter=',')
        target_writer.writerow(['sequence', 'motif_loci_importance', 'data_total:' + str(correct_predict_size)])

        for i in tqdm(range(correct_predict_size), desc="Data writing progress"):
            target_writer.writerow(post_perturbation_data[i])

    print('finished')


if __name__ == "__main__":
    main()
