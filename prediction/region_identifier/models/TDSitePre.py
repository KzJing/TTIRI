import torch
import torch.nn as nn
# from transformers import AutoModel, AutoConfig
# from transformers.models.bert.modeling_bert import BertPreTrainedModel

from .classifier.bert_classifier import BertForTokenClassification, BertForSequenceClassification
from .DNABERT2_MIX.bert_layers import BertModel
from .basic_module.merge_tokens import Merge_tokens
from .basic_module.bilstm import BiLSTM



class TDSitePre(nn.Module):  # BertPreTrainedModel  nn.Module

    def __init__(self, extra_config):
        super(TDSitePre, self).__init__()
        print("-----Initializing TDSitePre-----")

        if extra_config.classifier_order == 'serial':  # serial must correspond to seq_token
            if extra_config.predicting_content != 'seq_token':
                raise ValueError('The predicting_content should be seq_token if classifier_order is serial.')
                # config.predicting_content = 'seq_token'
                # print warning info

        self.base_frame = extra_config.base_frame
        self.num_merge_token = extra_config.num_merge_token
        self.merge_token_gap = extra_config.merge_token_gap
        self.keep_tail_tokens = extra_config.keep_tail_tokens
        self.merge_method = extra_config.merge_method
        self.post_mm_method = extra_config.post_mm_method

        self.predicting_content = extra_config.predicting_content
        self.seq_loss_coefficient = extra_config.seq_loss_coefficient

        if self.base_frame == 'bert':
            self.bf_dnabert2 = BertModel.from_pretrained(extra_config.model_name_or_path, add_pooling_layer=False)
        elif self.base_frame == 'bilstm':
            input_size = 768
            num_layers = 12
            self.bf_bilstm = BiLSTM(input_size=input_size,
                                 hidden_size=input_size,
                                 num_layers=num_layers,
                                 output_size=input_size,
                                 seq_level_output=False)
        else:
            raise ValueError('Invalid base frame for {}'.format(self.base_frame))

        self.classifier_type = extra_config.classifier
        if self.classifier_type == 'bert_token':
            self.classifier_for_tok = BertForTokenClassification.from_pretrained(
                extra_config.model_name_or_path,
                extra_config=extra_config
            )

        if self.predicting_content != 'token':
            self.classifier_for_seq = BertForSequenceClassification.from_pretrained(
                extra_config.model_name_or_path,
            )

        self.classifier_order = extra_config.classifier_order
        self.merge_tokens = Merge_tokens(num_merge_token=self.num_merge_token,
                                         strides=self.merge_token_gap,
                                         merge_method=self.merge_method,
                                         post_mm_method=self.post_mm_method,
                                         keep_tail_tokens=self.keep_tail_tokens)



    def forward(self, input_ids, attention_mask, labels):
        # input_ids: (bs, nt), nt == max_length

        if self.base_frame == 'bert':
            # dnabert2 is used to obtain the embedding representation
            bert_output = self.bf_dnabert2(input_ids=input_ids, attention_mask=attention_mask)
            hidden_logit = bert_output[0]  # : (b s h) = (bs nt h)
        elif self.base_frame == 'bilstm':
            # TODO: To be implemented, the input ids tags are converted to the embedded representation
            hidden_logit = self.bf_bilstm(input_ids)

        sequence_logit = hidden_logit[:, :1, :]  # The first token is the cls sequence classification token, (b, 1, h).
        sequence_labels = labels[:, :1]  # The dimension becomes (b, 1).

        token_logit = hidden_logit[:, 1:, :]  # The dimension becomes (b, s-1, h)
        token_labels = labels[:, 1:]  # The dimension becomes (b, s-1)

        if self.classifier_order == 'serial':
            return self.classifier_order_serial(sequence_logit, sequence_labels, token_logit, token_labels)

        elif self.classifier_order == 'parallel':
            return self.classifier_order_parallel(sequence_logit, sequence_labels, token_logit, token_labels)

        raise ValueError(f"Classifier_order not supported for {self.classifier_order}.")
        # loss = None
        # return loss, (None, None, None, None)  # loss, (seq_logit, seq_labels, tok_logit, tok_labels)


    def classifier_order_serial(self, sequence_logit, sequence_labels, token_logit, token_labels):
        seq_loss, seq_logit = self.classifier_for_seq(input_ids=sequence_logit, labels=sequence_labels)
        activation_logit = torch.sigmoid(seq_logit).view(-1)  # : (b 1 1) -> (b)
        mask_whether_pre_tok = activation_logit >= 0.5  # : (b)

        # Continue with the token prediction for that sequence
        merge_tok_logit, merge_tok_labels = self.merge_tokens(input_ids=token_logit, labels=token_labels)
        # merge_tok_logit: (b new_nt, h); merge_tok_labels: (b new_nt)

        # The element of dimension 0 is selected using the mask to filter out
        # the sequences predicted as distortion by seq for the next token prediction
        masked_token_logit = merge_tok_logit[mask_whether_pre_tok]  # (b new_nt, h) -> (new_b, new_nt, h)
        masked_token_labels = merge_tok_labels[mask_whether_pre_tok]  # (b new_nt) -> (new_b, new_nt)

        bs, nnt, _ = merge_tok_logit.shape
        tok_dtype = merge_tok_logit.dtype
        tok_device = merge_tok_logit.device
        wrap_tok_logit = torch.empty((bs, nnt, 1), dtype=tok_dtype, device=tok_device)  # : (b new_nt, 1)

        loss = seq_loss
        # token prediction is performed when there is a seq predicted to be distorted
        if mask_whether_pre_tok.sum(dim=0) > 0:
            tok_loss, tok_logit = self.classifier_for_tok(input_ids=masked_token_logit, labels=masked_token_labels)
            # tok_logit: (new_b, new_nt, 1)
            if tok_loss is not None:
                # loss = (seq_loss + tok_loss) / 2
                loss = self.seq_loss_coefficient * seq_loss + (1.0 - self.seq_loss_coefficient) * tok_loss

            wrap_tok_logit[mask_whether_pre_tok] = tok_logit.float()
        return loss, (seq_logit, sequence_labels, wrap_tok_logit, merge_tok_labels)

    def classifier_order_parallel(self, sequence_logit, sequence_labels, token_logit, token_labels):
        if self.predicting_content == 'sequence':
            seq_loss, seq_logit = self.classifier_for_seq(input_ids=sequence_logit, labels=sequence_labels)
            return seq_loss, (seq_logit, sequence_labels)

        # Extract tokens and combine them with prediction
        merge_tok_logit, merge_tok_labels = self.merge_tokens(input_ids=token_logit, labels=token_labels)
        tok_loss, tok_logit = self.classifier_for_tok(input_ids=merge_tok_logit, labels=merge_tok_labels)

        if self.predicting_content == 'seq_token':  # Sequence classification prediction is also required at this time
            seq_loss, seq_logit = self.classifier_for_seq(input_ids=sequence_logit, labels=sequence_labels)
            loss = seq_loss

            if tok_loss is not None:
                # loss = (seq_loss + tok_loss) / 2
                loss = self.seq_loss_coefficient * seq_loss + (1.0 - self.seq_loss_coefficient) * tok_loss
            logit = torch.cat([seq_logit, tok_logit], dim=1)
            labels = torch.cat([sequence_labels, merge_tok_labels], dim=1)
            return loss, (logit, labels)

        # Returns here when only predicting tokens
        return tok_loss, (tok_logit, merge_tok_labels)
