import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        bs, seq_len, hs = x.size()

        Q = self.query(x)  # (bs, seq_len, hidden_size)
        K = self.key(x)  # (bs, seq_len, hidden_size)
        V = self.value(x)  # (bs, seq_len, hidden_size)

        Q = Q.view(bs, seq_len, self.num_heads, self.head_dim)  # (bs, seq_len, num_heads, head_dim)
        K = K.view(bs, seq_len, self.num_heads, self.head_dim)  # (bs, seq_len, num_heads, head_dim)
        V = V.view(bs, seq_len, self.num_heads, self.head_dim)  # (bs, seq_len, num_heads, head_dim)

        # (bs, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))  # (bs, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # (bs, num_heads, seq_len, seq_len)

        out = torch.matmul(attention_weights, V)  # (bs, num_heads, seq_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.hidden_size)  # (bs, seq_len, hidden_size)

        out = self.fc_out(out)  # (bs, seq_len, hidden_size)

        return out, attention_weights


# # example
# if __name__ == "__main__":
#     bs = 32
#     seq_len = 50
#     hs = 128
#     num_heads = 8
#
#     x = torch.rand(bs, seq_len, hs)
#     self_attention = SelfAttention(hidden_size=hs, num_heads=num_heads)
#
#     out, attention_weights = self_attention(x)
#     print(out.shape)
#     print(attention_weights.shape)
