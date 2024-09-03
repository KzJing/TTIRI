import torch
import torch.nn as nn


class WBCELoss(nn.Module):

    def __init__(self, use_dynamic_pos_weight=True, static_pos_weight=None, mask_value=None, eps=1e-8,
                 only_consider_pos=False):
        super(WBCELoss, self).__init__()

        # By default, dynamic weights are used
        if use_dynamic_pos_weight is False and static_pos_weight is not None:
            self.use_dynamic_pos_weight = False
            self.static_pos_weight = static_pos_weight
        else:
            self.use_dynamic_pos_weight = True

        self.mask_value = mask_value
        self.eps = eps
        self.only_consider_pos = only_consider_pos


    def forward(self, logits, labels, total_mask_labels=None):
        # logits: (b s 1)
        # labels: (b s)

        loss = self.half_self_implement(logits, labels, total_mask_labels)

        return loss


    def half_self_implement(self, logits, labels, total_mask_labels=None):
        # logits: (b s 1)
        # labels: (b s)
        batch_size, num_tokens, _ = logits.shape
        bs_loss = []
        for i in range(batch_size):
            single_seq_tok_logits = logits[i]  # (s 1)
            single_seq_tok_labels = labels[i]  # (s)

            if total_mask_labels is None:
                mask_labels = single_seq_tok_labels != self.mask_value
            else:
                mask_labels = total_mask_labels[i]
            # Get the total number of tokens in each sample
            total_elements = mask_labels.sum(dim=0)  # : (s) -> (1)
            # The number of infidelity tokens in each sample is obtained
            n_pos = (single_seq_tok_labels * mask_labels).sum(dim=0)  # : (s) -> (1)

            # When only positive sequences are considered and the current sequence has no positive tokens,
            # the loss of the current sequence tokens is not calculated
            if self.only_consider_pos and n_pos == 0:
                continue
            if n_pos == 0:
                n_pos = 1

            pos_weight = (total_elements - n_pos) / n_pos  # : (1)

            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Assign a default weight to the positive samples
            loss = loss_fct(single_seq_tok_logits.view(-1)[mask_labels],
                            single_seq_tok_labels.float().view(-1)[mask_labels])
            bs_loss.append(loss.unsqueeze(0))  # : 1 -> (1)

        loss = None
        if len(bs_loss) > 0:
            loss = torch.cat(bs_loss, dim=0).mean(dim=0)  # : (b) -> (1)
        return loss

