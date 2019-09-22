import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.LSTM import Question_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class ProtocolNet(nn.Module):
    def __init__(self, fc_n_feature, mode='LSTM'):
        super(ProtocolNet, self).__init__()
        self.model = mode
        self.slide_window_size = 480

        self.in_dim = 1024 # hard coded for now...
        self.hidden_dim = 500
        self.out_dim = 1

        self.FC = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(self.hidden_dim, self.out_dim)).to(device)

        self.Q_LSTM = Question_LSTM().to(device)
        self.A_LSTM = Question_LSTM().to(device)
        self.embeds_QA = nn.Embedding(int(self.vocab_size), 300).to(device)
        d_model = 512  # Original res feature is 512d

        output_size = 512
        if self.model != 'QA':
            self.vq_fusion = nn.Sequential(nn.Linear(fc_n_feature-512, output_size), # hard coded 512...
                                 nn.ReLU(),
                                 nn.Dropout(0.3)).to(device)


    def forward(self, img_feat, q, choices, ans_idx, encoder=None):
        pass

    # ===== Multiple heads =====
    def multiheads(self, VQ_feature, Q_feature, choices, ans_idx):
        batch_size = Q_feature.size(0)
        n_choices = choices.size(1)

        if self.training:
            # permute the answer order during training (we keep it same for each batch for now)
            idx_permute = torch.randperm(n_choices)
            target = VQ_feature.new(batch_size, n_choices).fill_(0)
            target[:, (idx_permute == 0)] = 1
            embeded_A = self.embeds_QA(choices[:, idx_permute].long().to(device))
        else:
            # the correct answer is always at the begining for evaluation
            target = VQ_feature.new(batch_size, n_choices).fill_(0) # dummy
            embeded_A = self.embeds_QA(self.mat_a_sentence.long().to(device))

        A_feature = self.A_LSTM(embeded_A.view(-1, embeded_A.size(2), embeded_A.size(3)), \
            ).squeeze().view(embeded_A.size(0), 5, -1)

        if self.model == 'QA':
            cat_feature = torch.cat([Q_feature.unsqueeze(1).expand_as(A_feature), A_feature], dim=-1)
        else:
            cat_feature = torch.cat([VQ_feature.unsqueeze(1).expand_as(A_feature), A_feature], dim=-1)

        outs = self.FC(cat_feature).squeeze()

        return outs, target
