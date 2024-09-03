import json
import os
import csv
from typing import Dict, Sequence

import numpy as np
import torch
from dataclasses import dataclass, field
import transformers
from torch.utils.data import Dataset


'''
Assumed data format:
DNA sequence, sum_vars, infidelity_distribution, label
'''


class InfidelitySiteSamples(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        assert len(input_ids) == len(attention_mask) == len(labels)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], labels=self.labels[i])


def get_tokenizer_token(dna_seq, tokenizer, max_length):
    if max_length is None:
        padding_type = 'longest'
    else:
        padding_type = 'max_length'

    token_feat = tokenizer.batch_encode_plus(
        dna_seq,
        max_length=max_length,
        return_tensors='pt',
        padding=padding_type,  # longest max_length
        truncation=True
    )
    return token_feat


def prepare_infidelity_token_input(dna_seq, labels, tokenizer, max_length=None):
    dna_seq_token = get_tokenizer_token(dna_seq, tokenizer, max_length)
    input_ids = dna_seq_token['input_ids']
    attention_mask = dna_seq_token["attention_mask"]

    # Counting the number of valid tokens
    num_valid_tokens = []

    # Calculate the corresponding token_labels
    # Get the tokenized dictionary
    vocab_dict = tokenizer.get_vocab()
    # Swap keys and values to get a new dictionary
    inverted_vocab_dict = {v: k for k, v in vocab_dict.items()}

    # Converting raw transcript infidelity labels into token labels relative to the BPE dictionary
    token_labels = []
    # e.g input_ids[i]: [1, t1, t2, t3, t4, ..., tk, 2, 0, 0, ..., 0] where len(input_ids[i]) == longest
    for i, sample_input_ids in enumerate(input_ids):  # Iteration sample
        sample_labels = [-100] * input_ids.shape[-1]  # An initialized label of -100 indicates an invalid label.
        sample_labels[0] = labels[i][0]  # Retaining the cls token
        # Count the number of valid tokens
        num_vt = 0

        s = 0  # 开始下标
        for j, ids in enumerate(sample_input_ids):  # Iterate over the tokens within the sample
            if ids >= 5:  # Handling of non-special tokens only
                # Counting the number of valid tokens
                num_vt += 1

                try:
                    # Query the length of the base sequence corresponding to ids according to the dictionary
                    length = len(inverted_vocab_dict[int(ids)])  # Convert tensor to int
                except KeyError:
                    # Handle the case when the key doesn't exist
                    print(f"Key {ids} does not exist in the dictionary.")
                flag = False  # Indicates that there are no infidelity sites in the token

                for base_label in labels[i][s: s + length]:
                    if base_label == 1:  # Indicates that the token is infidelity
                        flag = True
                        break
                s = s + length
                # The currently valid token is marked with a 1 to indicate the presence of infidelity and a 0 to indicate the opposite.
                sample_labels[j] = 1 if flag else 0

        num_valid_tokens.append(num_vt)
        token_labels.append(sample_labels)

    # Output/return the number of valid tokens counted.
    max_num = max(num_valid_tokens)
    min_num = min(num_valid_tokens)
    print(f"max_num：{max_num}")
    print(f"min_num：{min_num}")

    num_valid_tokens_array = np.array(num_valid_tokens)
    # Calculation of the median
    median_value = np.median(num_valid_tokens_array)
    mean_value = np.mean(num_valid_tokens_array)
    print(f"The median value is: {median_value}")
    print(f"The mean value is: {mean_value}")

    return input_ids, attention_mask, token_labels


def infidelity_site_loader_csv(args, tokenizer, load_type='train', delimiter=','):
    assert load_type in ['train', 'val', 'test']
    if load_type == 'train':
        data_path = os.path.join(args.data_path, args.train_data_name)
    elif load_type == 'val':
        data_path = os.path.join(args.data_path, args.val_data_name)
    else:
        data_path = os.path.join(args.data_path, args.test_data_name)

    with open(data_path) as csvfile:
        data = list(csv.reader(csvfile, delimiter=delimiter))[1:]  # Remove header line

    dna_seq = [d[0] for d in data]

    # Each site was recorded as infidelity or not, with 0 indicating normal and 1 indicating infidelity.
    source_labels = []
    for d in data:
        site_infidelity_label = json.loads(d[2])
        if int(d[3]) == 0:  # Negative samples, then the site label is all zeros
            source_labels.append([0] + [0] * len(site_infidelity_label))
        else:  # When there is infidelity in the sample
            source_labels.append([1] + [0 if tag == 0 else 1 for tag in site_infidelity_label])

    # Tokenizing the sequence
    input_ids, attention_mask, token_labels = prepare_infidelity_token_input(dna_seq, source_labels, tokenizer, max_length=args.max_length)
    dataset = InfidelitySiteSamples(input_ids, attention_mask, token_labels)

    return dataset


@dataclass
class DataCollatorForInfidelityDataset(object):
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

