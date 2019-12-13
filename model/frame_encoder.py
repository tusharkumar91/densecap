import torch.nn as nn
import torch
from torch.autograd import Variable
from model.transformer import Transformer

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


class FrameEncoder(torch.nn.Module):
    def __init__(self, d_model, in_emb_dropout, d_hidden, n_layers, n_heads, attn_dropout):
        super(FrameEncoder, self).__init__()
        #self.rgb_emb = nn.Linear(2048, d_model // 2)
        #self.flow_emb = nn.Linear(1024, d_model // 2)
        #self.emb_out = nn.Sequential(
        #    # nn.BatchNorm1d(h_dim),
        #    DropoutTime1D(in_emb_dropout),
        #    nn.ReLU()
        #)

        self.vis_emb = Transformer(d_model, 0, 0,
                                   d_hidden=d_hidden,
                                   n_layers=n_layers,
                                   n_heads=n_heads,
                                   drop_ratio=attn_dropout)

    def forward(self, x, mask=None):
        #x_rgb, x_flow = torch.split(x, 2048, 2)
        #x_rgb = self.rgb_emb(x_rgb.contiguous())
        #x_flow = self.flow_emb(x_flow.contiguous())

        #x = torch.cat((x_rgb, x_flow), 2)

        #x = self.emb_out(x)

        vis_feat, all_emb = self.vis_emb(x, mask)
        return vis_feat, all_emb, x
