import torch
from torch import nn

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
# input_size = 512  # rnn input size / image width
word_dim = 300
LR = 0.01  # learning rate

class Video_LSTM(nn.Module):
    def __init__(self, hidden_size=1000, mode='normal'):
        super(Video_LSTM, self).__init__()

        if mode=='C':
            input_size = 768
        elif mode=='ATT_D' or mode=='ATT_C' :
            input_size = 1024
        elif mode=='ATT_DC':
            input_size = 1024+512
        elif mode=='TREE_C':
            input_size = 256
        else:
            input_size = 512
        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,
            # input & output will have batch size as first dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x, mask=None):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)

        r_out, _ = self.rnn(x, None)  # None means no initial hidden state
        # choose r_out at the last time step
        if mask is not None:
            out = []
            for i in range(r_out.size(0)):
                out.append(r_out[i, mask[i]-1]) # mask on the last frame of each video
            out = torch.stack(out)
        else:
            out = r_out[:, -1, :]

        return out


class Question_LSTM(nn.Module):
    def __init__(self, input_size=300, hidden_size=1024):
        super(Question_LSTM, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,
            bidirectional=True
            # input & output will have batch size as first dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x, mask=None):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)

        r_out, _ = self.rnn(x, None)  # None means no initial hidden state
        # choose r_out at the last time step
        if mask is not None:
            out = []
            for i in range(r_out.size(0)):
                out.append(r_out[i, mask[i]-1]) # mask on the last frame of each sentence
            out = torch.stack(out)
        else:
            out = r_out[:, -1, :]
        return out

class LSTM_cell(nn.Module):
    def __init__(self, hidden_size=1000, mode='normal'):
        super(LSTM_cell, self).__init__()

        if mode == 'C':
            input_size = 768
        elif mode == 'ATT_D':
            input_size = 1024
        elif mode == 'ATT_C':
            input_size = 768
        elif mode == 'ATT_DC':
            input_size = 1024 + 512
        elif mode == 'TREE_C':
            input_size = 256
        elif mode in ['LATE_FUSION_D','LATE_FUSION_C']:
            input_size = 768
        else:
            input_size = 512

        self.rnn = nn.GRUCell(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,  # rnn hidden unit

            # input & output will have batch size as first dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, input, hx):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        hx = self.rnn(input,hx)
        return hx
