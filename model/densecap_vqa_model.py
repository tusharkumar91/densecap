import torch
import time
import numpy as np
from torch import nn
from model.ProtocolNet import ProtocolNet
from model.transformer import Transformer
from model.frame_encoder import FrameEncoder
import sys
from torch.autograd import Variable
import torch.nn.functional as F


sys.path.append('.')

fc_dict = {'QA': 1024, 'LSTM': 2024, 'MAC': 2560}

def positional_encodings(x, D):
    # input x a vector of positions
    encodings = torch.zeros(x.size(0), D)
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    encodings = Variable(encodings)

    for channel in range(D):
        if channel % 2 == 0:
            encodings[:,channel] = torch.sin(
                x / 10000 ** (channel / D))
        else:
            encodings[:,channel] = torch.cos(
                x / 10000 ** ((channel - 1) / D))
    return encodings

class DropoutTime1D(nn.Module):
    '''
        assumes the first dimension is batch,
        input in shape B x T x H
        '''
    def __init__(self, p_drop):
        super(DropoutTime1D, self).__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if self.training:
            mask = x.data.new(x.data.size(0),x.data.size(1), 1).uniform_()
            mask = Variable((mask > self.p_drop).float())
            return x * mask
        else:
            return x * (1-self.p_drop)

    def init_params(self):
        pass

    def __repr__(self):
        repstr = self.__class__.__name__ + ' (\n'
        repstr += "{:.2f}".format(self.p_drop)
        repstr += ')'
        return repstr

class TransformerBaseline(ProtocolNet):
    def __init__(self, vocab_size, mode='Trans', d_model=1024, in_emb_dropout=0.1):
        self.name = mode
        self.vocab_size = vocab_size
        fc_n_feature = 1024
        super(TransformerBaseline, self).__init__(1024 + 512, mode)
        self.linear = nn.Sequential(nn.Linear(1024, 512))
        self.frame_emb = FrameEncoder(d_model, in_emb_dropout, 2048, 2, 8, 0.2)
        # self.rgb_emb = nn.Linear(2048, d_model // 2)
        # self.flow_emb = nn.Linear(1024, d_model // 2)
        # self.emb_out = nn.Sequential(
        #     # nn.BatchNorm1d(h_dim),
        #     DropoutTime1D(in_emb_dropout),
        #     nn.ReLU()
        # )
        #
        # self.vis_emb = Transformer(d_model, 0, 0,
        #                            d_hidden=2048,
        #                            n_layers=2,
        #                            n_heads=8,
        #                            drop_ratio=0.2)


    def forward(self, img_feat, q, choices, ans_idx):
        # print("forward called")
        tic = time.time()
        tok = time.time()
        # print(question)
        embeded_Q = self.embeds_QA(q.long().cuda())
        Q_feature = self.Q_LSTM(embeded_Q)

        tic = time.time()
        V_feature, all_emb, _ = self.frame_emb(img_feat)
        V_feature = self.linear(V_feature)
        V_feature = nn.MaxPool1d(kernel_size=V_feature.shape[1])(V_feature.view(V_feature.size(0), 512, -1)).squeeze(2)
        tok = time.time()
        VQ_feature = torch.cat([nn.functional.layer_norm(V_feature, [V_feature.shape[-1]]),
                                nn.functional.layer_norm(Q_feature, [Q_feature.shape[-1]])], dim=1)
        VQ_feature = self.vq_fusion(VQ_feature)
        outs = self.multiheads(VQ_feature, Q_feature, choices, ans_idx)
        return outs
