import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_level_output=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_level_output = seq_level_output

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        hidden_times = 4 if seq_level_output else 2

        self.fc = nn.Linear(hidden_size * hidden_times, output_size)

    def forward(self, x):
        # x : (bs, seq_len, hs)

        out, _ = self.lstm(x)

        if self.seq_level_output:
            out = torch.cat((out[:, 0, :], out[:, -1, :]), dim=1)  # : (bs, hs * 4)

        out = self.fc(out)  # : (bs, seq_len, hs) or (bs, hs)
        return out
