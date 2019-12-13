__author__ = "Jie Lei"

import torch
from torch import nn
import random
from rnn import RNNEncoder, max_along_time
from bidaf import BidafAttn
from mlp import MLP


class ABC(nn.Module):
    def __init__(self, vocab_size):
        super(ABC, self).__init__()
        self.vid_flag = True
        self.sub_flag = False
        self.vcpt_flag = False
        hidden_size_1 = 150
        hidden_size_2 = 300
        n_layers_cls = 1
        vid_feat_size = 3072
        embedding_size = 300
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size_1 * 3, method="dot")  # no parameter for dot
        self.lstm_raw = RNNEncoder(300, hidden_size_1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        self.seq_pred = nn.Sequential(nn.Linear(900, 256), nn.Tanh(), nn.Linear(256, 2)).cuda()
        
        if self.vid_flag:
            print("activate video stream")
            self.video_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(vid_feat_size, embedding_size),
                nn.Tanh(),
            )
            self.lstm_mature_vid = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vid = MLP(hidden_size_2*2, 1, 500, n_layers_cls)
        '''
        if self.sub_flag:
            print("activate sub stream")
            self.lstm_mature_sub = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_sub = MLP(hidden_size_2*2, 1, 500, n_layers_cls)
        
        #if self.vcpt_flag:
        #print("activate vcpt stream")
        self.lstm_mature_vcpt = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                           dropout_p=0, n_layers=1, rnn_type="lstm")
        self.classifier_vcpt = MLP(hidden_size_2*2, 1, 500, n_layers_cls)
        self.label_lstm = RNNEncoder(hidden_size_2, hidden_size_2, bidirectional=False,
                                     dropout_p=0, n_layers=1, rnn_type="lstm")
        '''

    #def load_embedding(self, pretrained_embedding):
    #    self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, img, vid, vid_l, q, q_l, ch, ch_l, num_seg):
        idx_permute = torch.randperm(5)
        vid = torch.zeros((vid.shape)).cuda()
        target = torch.zeros((q.size(0), 5)).long().cuda()
        target[:, (idx_permute == 0)] = 1
        ch = ch[:, idx_permute]
        ch_l = ch_l[:, idx_permute] 
        
        a0 = ch[:, 0]
        a1 = ch[:, 1]
        a2 = ch[:, 2]
        a3 = ch[:, 3]
        a4 = ch[:, 4]
        a0_l = ch_l[:, 0]
        a1_l = ch_l[:, 1]
        a2_l = ch_l[:, 2]
        a3_l = ch_l[:, 3]
        a4_l = ch_l[:, 4]
        e_q = self.embedding(q)
        e_a0 = self.embedding(a0)
        e_a1 = self.embedding(a1)
        e_a2 = self.embedding(a2)
        e_a3 = self.embedding(a3)
        e_a4 = self.embedding(a4)
        
        raw_out_q, _ = self.lstm_raw(e_q, q_l.reshape((-1)))
        raw_out_a0, _ = self.lstm_raw(e_a0, a0_l)
        raw_out_a1, _ = self.lstm_raw(e_a1, a1_l)
        raw_out_a2, _ = self.lstm_raw(e_a2, a2_l)
        raw_out_a3, _ = self.lstm_raw(e_a3, a3_l)
        raw_out_a4, _ = self.lstm_raw(e_a4, a4_l)
        #print(e_a4.shape, a4_l.shape)
        if self.sub_flag:
            e_sub = self.embedding(sub)
            raw_out_sub, _ = self.lstm_raw(e_sub, sub_l)
            sub_out = self.stream_processor(self.lstm_mature_sub, self.classifier_sub, raw_out_sub, sub_l,
                                            raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                            raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            sub_out = 0

        if self.vcpt_flag:
            #vid = vid.reshape((vid.size(0), -1))
            #vcpt = torch.zeros((vid.size(0), 340)).long().cuda()
            #vid_counts = torch.zeros((vid.size(0))).long().cuda()
            embeded_labels = self.embedding(vid.long().cuda())
            label_feat = torch.zeros((vid.size(0), 17, 300)).cuda()
            vcpt_l = vid_l
            for i in range(vid.size(0)):
                label_feature_i = torch.zeros((17, 300)).cuda()
                for j in range(vcpt_l[i]):
                    #print(embeded_labels[i, j].unsqueeze(0).shape, torch.LongTensor([20]))
                    label_feature_j_out, _ = self.lstm_raw(embeded_labels[i, j].unsqueeze(0), torch.LongTensor([20]).cuda())
                    label_feature_j_out = label_feature_j_out[:, -1, :]
                    label_feature_i[j] = label_feature_j_out.reshape(-1).squeeze()
                label_feat[i] = label_feature_i
                #vid_u = torch.unique(vid[i]).squeeze()
                #count = vid_u.size(0)
                #vid_counts[i] = count
                #vcpt[i, :count] = vid_u
            #vcpt_l = vid_counts
            e_vcpt = label_feat
            #e_vcpt = self.embedding(vcpt)
            raw_out_vcpt, _ = self.label_lstm(e_vcpt, vcpt_l)
            #print(raw_out_vcpt.shape, vcpt_l)
            #exit(0)
            vcpt_out = self.stream_processor(self.lstm_mature_vcpt, self.classifier_vcpt, raw_out_vcpt, vcpt_l,
                                             raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                             raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vcpt_out = 0

        if self.vid_flag:
            e_vid = self.video_fc(vid)
            raw_out_vid, _ = self.lstm_raw(e_vid, vid_l)
            vid_out = self.stream_processor(self.lstm_mature_vid, self.classifier_vid, raw_out_vid, vid_l,
                                            raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                            raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vid_out = 0

        out = sub_out + vcpt_out + vid_out  # adding zeros has no effect on backward
        '''
        cls_input = []
        cls_output = []
        input_idx = 0
        for i in range(vid.size(0)):
            pairs = []
            pairs = [(k,k+1) for k in range(vid_l[i]-1)]
            random.shuffle(pairs)
            pairs = pairs[:3]
            #before 0 after 1
            vid_info = max_along_time(raw_out_vcpt[i].unsqueeze(0), vcpt_l[i].unsqueeze(0)).squeeze()
            for pair in pairs:
                p = random.random()
                #print(raw_out_vcpt[i].shape, label_feat[i, 0].shape)
                if p < 0.5:
                    x, y = pair
                    label = 0
                    #cls_input[input_idx] = torch.cat([kb[i], label_feat_kb[i, x], label_feat_kb[i, y]])
                    cls_input.append(torch.cat([vid_info, label_feat[i, x], label_feat[i, y]]).unsqueeze(0))
                else:
                    y, x = pair
                    label = 1
                    #cls_input[input_idx] = torch.cat([kb[i], label_feat_kb[i, y], label_feat_kb[i, x]])
                    cls_input.append(torch.cat([vid_info, label_feat[i, y], label_feat[i, x]]).unsqueeze(0))
                cls_output.append(label)
                input_idx += 1
        cls_input = torch.cat(cls_input, dim=0).cuda()
        cls_output = torch.LongTensor(cls_output).cuda()
        #print(cls_input.shape, cls_output.shape)
        cls_pred = self.seq_pred(cls_input)
        '''
        cls_pred, cls_output = (torch.rand((32, 5)).cuda(), torch.ones((32, 5)).long().cuda())
        return (out.squeeze(), torch.argmax(target, 1)), (cls_pred, cls_output)

    def stream_processor(self, lstm_mature, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):
        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)

        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)

        mature_maxout_a0, _ = lstm_mature(concat_a0, ctx_l)
        mature_maxout_a1, _ = lstm_mature(concat_a1, ctx_l)
        mature_maxout_a2, _ = lstm_mature(concat_a2, ctx_l)
        mature_maxout_a3, _ = lstm_mature(concat_a3, ctx_l)
        mature_maxout_a4, _ = lstm_mature(concat_a4, ctx_l)

        mature_maxout_a0 = max_along_time(mature_maxout_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = max_along_time(mature_maxout_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = max_along_time(mature_maxout_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = max_along_time(mature_maxout_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = max_along_time(mature_maxout_a4, ctx_l).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out

    @staticmethod
    def get_fake_inputs(device="cuda:0"):
        bsz = 16
        q = torch.ones(bsz, 25).long().to(device)
        q_l = torch.ones(bsz).fill_(25).long().to(device)
        a = torch.ones(bsz, 5, 20).long().to(device)
        a_l = torch.ones(bsz, 5).fill_(20).long().to(device)
        a0, a1, a2, a3, a4 = [a[:, i, :] for i in range(5)]
        a0_l, a1_l, a2_l, a3_l, a4_l = [a_l[:, i] for i in range(5)]
        sub = torch.ones(bsz, 300).long().to(device)
        sub_l = torch.ones(bsz).fill_(300).long().to(device)
        vcpt = torch.ones(bsz, 300).long().to(device)
        vcpt_l = torch.ones(bsz).fill_(300).long().to(device)
        vid = torch.ones(bsz, 100, 2048).to(device)
        vid_l = torch.ones(bsz).fill_(100).long().to(device)
        return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l


if __name__ == '__main__':
    from config import BaseOptions
    import sys
    sys.argv[1:] = ["--input_streams" "sub"]
    opt = BaseOptions().parse()

    model = ABC(opt)
    model.to(opt.device)
    test_in = model.get_fake_inputs(device=opt.device)
    test_out = model(*test_in)
    print(test_out.size())
