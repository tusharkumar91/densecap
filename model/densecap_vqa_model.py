import torch
import time
import numpy as np
from torch import nn
from model.ProtocolNet import ProtocolNet
import sys
import torch.nn.functional as F

sys.path.append('.')

fc_dict = {'QA': 1024, 'LSTM': 2024, 'MAC': 2560}


class TransformerBaseline(ProtocolNet):
    def __init__(self, mode='Trans', fusion_mode='Embed'):
        self.name = mode
        fc_n_feature = 1024
        super(TransformerBaseline, self).__init__(1024 + 512, mode, fusion_mode)
        self.linear = nn.Sequential(nn.Linear(1024, 512))


    def forward(self, img_feat, q, choices, ans_idx, encoder=None):
        # print("forward called")
        tic = time.time()
        tok = time.time()
        # print(question)
        embeded_Q = self.embeds_QA(q.long().cuda())
        Q_feature = self.Q_LSTM(embeded_Q)

        tic = time.time()
        V_feature = encoder(img_feat)
        V_feature = self.linear(V_feature)
        V_feature = nn.MaxPool1d(kernel_size=V_feature.shape[1])(V_feature.view(V_feature.size(0), 512, -1)).squeeze(2)
        tok = time.time()
        VQ_feature = torch.cat([nn.functional.layer_norm(V_feature, [V_feature.shape[-1]]),
                                nn.functional.layer_norm(Q_feature, [Q_feature.shape[-1]])], dim=1)
        VQ_feature = self.vq_fusion(VQ_feature)
        outs = self.multiheads(VQ_feature, Q_feature, choices, ans_idx)
        return outs
