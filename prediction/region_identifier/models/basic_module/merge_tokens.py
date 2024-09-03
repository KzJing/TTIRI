import torch
import torch.nn as nn
from .bilstm import BiLSTM


class Merge_tokens(nn.Module):

    def __init__(self, num_merge_token=1, strides=1, merge_method='mean', input_size=768, post_mm_method='none', keep_tail_tokens=False):
        super(Merge_tokens, self).__init__()
        self.num_merge_token = num_merge_token
        self.strides = strides

        assert merge_method in ('mean', 'max', 'mean_max', 'bilstm'), 'merge_method must be mean or bilstm.'
        self.merge_method = merge_method

        assert post_mm_method in ('none', 'bilstm'), 'post_mm_method must be none or bilstm.'
        self.post_mm_method = post_mm_method


        if self.merge_method == 'bilstm':
            self.bilstm = BiLSTM(input_size=input_size,
                                 hidden_size=input_size,
                                 num_layers=4,
                                 output_size=input_size,
                                 seq_level_output=True)

        if self.post_mm_method == 'bilstm':
            self.post_bilstm = BiLSTM(input_size=input_size,
                                      hidden_size=input_size,
                                      num_layers=4,
                                      output_size=input_size,
                                      seq_level_output=False)

        if self.merge_method == 'mean_max':
            self.mean_max_fc_out = nn.Linear(input_size * 2, input_size)

        self.keep_tail_tokens = keep_tail_tokens


    def forward(self, input_ids, labels):
        merge_tok_logits, merge_tok_labels = self.merge_tokens_sliding_window(input_ids, labels)

        if self.post_mm_method == 'bilstm':
            merge_tok_logits = self.post_bilstm(merge_tok_logits)

        return merge_tok_logits, merge_tok_labels


    def merge_tokens_sliding_window(self, input_ids, labels):
        '''
            This function implements the merging of m adjacent token numbers, denoted as m-tokens.

            When the requested merge number m is greater than the total number of tokens in the current sequence,
            we default to merge the current sequence to 1 new token.

            Finally, the newly merged logits and labels objects are populated to ensure that the dimensions
            of the data are consistent across batches (and, of course, across batches).

            Requirement: the only parameters passed in by default are the presence of a padded token at the end of the sequence.
            :param input_ids: (b s h)
            :param labels: (b s)
            :return:
        '''
        window_size = self.num_merge_token
        strides = self.strides

        device = input_ids.device

        batch_size, num_token, hidden_size = input_ids.shape

        mask_token = labels != -100  # (b, s) bool
        mask_token_expand = mask_token.unsqueeze(-1).expand_as(input_ids)  # (b, s, h) bool

        valid_input_ids = input_ids * mask_token_expand  # (b, s, h) float
        # Retain valid marker 0/1 and change invalid marker -100 to 0
        valid_labels = labels * mask_token  # (b s) long
        # Get the inverse token mask, where valid tokens correspond to 0/False and invalid tokens correspond to 1/True.
        # not_mask_token = ~mask_token
        # If you choose to keep the remaining tokens at the end, then the mask information is a valid mask, otherwise it is an invalid mask.
        mask_info = mask_token if self.keep_tail_tokens else ~mask_token

        if window_size > num_token:
            strides = num_token

        new_bs_logits = []
        new_bs_labels = []
        new_bs_not_mask = []

        begin_idx = 0
        end_idx = begin_idx + window_size  # index left closed right open

        while end_idx <= num_token:
            if self.merge_method == 'mean':
                mean_embedding = valid_input_ids[:, begin_idx:end_idx, :].mean(dim=1).unsqueeze(1)  # (b, 1, h)
                new_bs_logits.append(mean_embedding)
            elif self.merge_method == 'max':
                max_embedding, _ = valid_input_ids[:, begin_idx:end_idx, :].max(dim=1)  # (b, h)
                max_embedding = max_embedding.unsqueeze(1)  # (b, 1, h)
                new_bs_logits.append(max_embedding)  # append: (b, 1, h)
            elif self.merge_method == 'mean_max':
                mean_embedding = valid_input_ids[:, begin_idx:end_idx, :].mean(dim=1).unsqueeze(1)  # (b, 1, h)
                max_embedding, _ = valid_input_ids[:, begin_idx:end_idx, :].max(dim=1)  # (b, h)
                max_embedding = max_embedding.unsqueeze(1)  # (b, 1, h)
                new_bs_logits.append(torch.cat([mean_embedding, max_embedding], dim=-1))  # append: (b, 1, 2*h)
            elif self.merge_method == 'bilstm':
                new_bs_logits.append(self.bilstm(valid_input_ids[:, begin_idx:end_idx, :]).unsqueeze(1))  # append: (b, 1, h)
            new_bs_labels.append(valid_labels[:, begin_idx:end_idx].sum(dim=1).unsqueeze(1))  # append: (b, 1)
            new_bs_not_mask.append(mask_info[:, begin_idx:end_idx].sum(dim=1).unsqueeze(1))  # append: (b, 1)

            begin_idx += strides
            end_idx = begin_idx + window_size

        new_logits = torch.cat(new_bs_logits, dim=1)  # : (b ns h)
        if self.merge_method == 'mean_max':  # new_logits : (b ns 2*h)
            new_logits = self.mean_max_fc_out(new_logits)  # : (b ns h)

        merged_bs_labels = torch.cat(new_bs_labels, dim=1)  # : (b ns)
        merged_bs_mask_info = torch.cat(new_bs_not_mask, dim=1)  # : (b ns)

        pos_mask_mbs_labels = merged_bs_labels != 0
        if self.keep_tail_tokens:
            invalid_token_mask = merged_bs_mask_info == 0
        else:
            invalid_token_mask = merged_bs_mask_info != 0
        new_labels = torch.zeros_like(pos_mask_mbs_labels, device=device).long()
        new_labels[pos_mask_mbs_labels] = 1
        new_labels[invalid_token_mask] = -100

        # After merging the tokens of a sequence, there should be at least one valid token.
        mask_col_0 = new_labels[:, 0] == -100  # : (b)
        if mask_col_0.sum(dim=0) > 0:
            print('Warning: The number of merge tokens is too large, resulting in a complete and'
                  ' valid m-tokens that do not exist!')
            print('The force operation will now be taken to guarantee the existence of at least one'
                  ' valid token in a sequence, even if its number of valid tokens is less than m.')

            new_labels[:, 0] = pos_mask_mbs_labels[:, 0].long()

        return new_logits, new_labels

