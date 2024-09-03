import copy
import logging
import math
import sys
import warnings
from typing import List, Optional, Tuple, Union, Any

# from TorchCRF import CRF
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (MaskedLMOutput, SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import BertPreTrainedModel


from ..loss_function.wbce_loss import WBCELoss


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHeadForClassification(nn.Module):

    def __init__(self, config, last_dimension_size=1):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)  # Contains linear layers of the same dimension + activation (relu) + layer normalisation
        self.decoder = nn.Linear(config.hidden_size, last_dimension_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertForTokenClassification(BertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(self, config, extra_config):
        super(BertForTokenClassification, self).__init__(config)
        assert extra_config.loss_function in ('CE', 'WBCE', 'DL'), f"Invalid loss_function: {extra_config.loss_function}"
        assert extra_config.WBCE_type in ('batch_sequence_weighted', 'single_sequence_weighted',
                                          'ssw_only_consider_pos'), f"Invalid WBCE_type: {extra_config.WBCE_type}"

        # Loss parameter configuration
        self.loss_function = extra_config.loss_function
        self.smooth = extra_config.dl_params_smooth
        self.ohem_ratio = extra_config.dl_params_ohem_ratio
        self.alpha = extra_config.dl_params_alpha
        self.square_denominator = extra_config.dl_params_square_denominator

        self.WBCE_type = extra_config.WBCE_type
        self.use_dynamic_pos_weight = extra_config.bce_params_use_dynamic_pos_weight
        self.pos_weight = extra_config.bce_params_pos_weight

        # [DNA]BERT base configuration
        self.config = config

        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout_tok = nn.Dropout(classifier_dropout)

        if self.loss_function in ('WBCE', 'DL'):
            self.last_dimension_size = 1
        else:
            self.last_dimension_size = 2
        self.classifier_tok = BertLMPredictionHeadForClassification(config, last_dimension_size=self.last_dimension_size)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(self, input_ids, labels):
        sequence_output = input_ids
        sequence_output = self.dropout_tok(sequence_output)
        logits = self.classifier_tok(sequence_output)

        loss_function = self.loss_function

        loss = None
        if labels is not None:
            # Compute loss
            if loss_function == 'CE':
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            elif loss_function == 'WBCE':
                if self.WBCE_type == 'batch_sequence_weighted':
                    mask_labels = labels.view(-1) != -100  # Samples with label -100 are ignored
                    # Dynamic positive sample weights are used by default, using the inverse of the percentage of
                    # positive tokens among all tokens in the sequence of this batch as its dynamic weight.
                    if self.use_dynamic_pos_weight:
                        selected_labels = labels.view(-1)[mask_labels]
                        count_ones = torch.sum(selected_labels == 1)
                        if count_ones == 0:
                            count_ones = 1
                        total_elements = selected_labels.numel()
                        rec_ratio = (total_elements - count_ones) / count_ones
                    else:
                        rec_ratio = self.pos_weight
                    # Initialize BCEWithLogitsLoss with pos_weight
                    pos_weight = torch.tensor([rec_ratio], device=input_ids.device)
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fct(logits.view(-1)[mask_labels], labels.float().view(-1)[mask_labels])
                else:
                    if self.WBCE_type == 'ssw_only_consider_pos':
                        only_consider_pos = True
                    else:
                        only_consider_pos = False
                    loss_fct = WBCELoss(use_dynamic_pos_weight=self.use_dynamic_pos_weight,
                                        static_pos_weight=self.pos_weight,
                                        mask_value=-100,
                                        only_consider_pos=only_consider_pos)
                    loss = loss_fct(logits, labels)
            else:
                print('The loss calculation method is wrong, only one of (CE, BCE) can be selected')
                sys.exit(1)

        return loss, logits


class BertForSequenceClassification(BertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout_seq = nn.Dropout(classifier_dropout)
        self.classifier_seq = BertLMPredictionHeadForClassification(config, last_dimension_size=1)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(self, input_ids, labels):
        sequence_output = input_ids
        sequence_output = self.dropout_seq(sequence_output)
        logits = self.classifier_seq(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            labels_float = labels.float()
            loss = loss_fct(logits.view(-1), labels_float.view(-1))

        return loss, logits