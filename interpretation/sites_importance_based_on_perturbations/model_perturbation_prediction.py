'''

    This script is used to predict the data before and after the perturbation,
    here using the dnabert model, while storing the prediction results.

'''
import csv
import math
import pandas as pd
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


"""
Manually calculate the evaluation index.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)

    correct_predictions = [i for i in range(len(predictions)) if predictions[i] == labels[i]]

    # Get the log-odds of the correctly predicted sequences.
    log_odds = []
    extreme_val_cnt = 0
    for item in range(logits):
        total = item[0] + item[1]  # Obtain the probabilities that the prediction is positive and negative
        correct_probability = item[1] / total
        if correct_probability == 1.0 or correct_probability == 0.0:
            extreme_val_cnt += 1
        current_log_odds = calculate_log_odds(correct_probability)
        log_odds.append(current_log_odds)

    return {
        "logits": logits,
        "correct_predictions": correct_predictions,
        "log_odds": log_odds,
        "extreme_val_cnt": extreme_val_cnt,
    }


"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
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


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="DNABERT2_run")  # The name of the folder to output the file
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=50, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=256)
    per_device_eval_batch_size: int = field(default=256)
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


def get_prediction_log_odds(trainer, data, labels):
    test_dataset = SupervisedDataset(tokenizer=trainer.tokenizer, texts=data, labels=labels,
                                     num_labels=trainer.model.num_labels)
    outputs = trainer.evaluate(eval_dataset=test_dataset)
    log_odds = outputs['eval_log_odds']
    extreme_val_cnt = outputs['eval_extreme_val_cnt']
    return log_odds, extreme_val_cnt



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

    in_file_path = os.path.join(data_args.data_path, "dataset_PA.csv")

    data = []
    labels = []
    remain_data = []
    with open(in_file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)  # skip header
        for row in tqdm(reader, desc="Data read progress"):
            sequence = row[0]
            label = int(row[1])
            remain = row[2:]
            if label == 0:
                continue
            data.append(sequence)
            labels.append(label)
            remain_data.append(remain)

    log_odds, extreme_val_cnt = get_prediction_log_odds(trainer, data, labels)

    print(f'log odds Number of 0/1 or -inf/inf extremes:{extreme_val_cnt}')
    if len(data) == len(log_odds):
        print('The number of data is equal to the log odds of the result, correct!')
    else:
        print('The number of data is not equal to the number of log odds of the result, error!')

    out_file = os.path.join(data_args.data_path, "dataset_PA_post.csv")
    with open(out_file, 'w', newline='') as outfile:
        target_writer = csv.writer(outfile, delimiter=',')
        target_writer.writerow(header[0:2] + 'log_odds' + header[2:])

        for i in tqdm(range(len(data)), desc="Data writing progress"):
            target_writer.writerow(data[i] + label[i] + log_odds[i] + remain_data[i])

    print('finished')


if __name__ == "__main__":
    main()
