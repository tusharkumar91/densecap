import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.LSTM import Question_LSTM
from model.model import Net
from transformers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProtocolNet(nn.Module):
    def __init__(self, fc_n_feature, mode='LSTM', d_model=1024):
        super(ProtocolNet, self).__init__()
        self.model = mode
        self.slide_window_size = 480
        self.in_dim = 1024 # hard coded for now...
        self.hidden_dim = 500
        self.out_dim = 1
        #self.linear = nn.Sequential(nn.Linear(6144, 1024), nn.ReLU())
        #self.linear_v = nn.Sequential(nn.Linear(2048, 1024), nn.Tanh(), nn.Dropout(0.1))
        #self.base_emb = Net(video_dim=4096, embd_dim=6144, we_dim=300, max_words=36)
        #self.base_emb.load_state_dict(torch.load("/home/cxu-serve/p1/tusharku/howto100m/model/howto100m_pt_model.pth"))
        #self.FC = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim),
        #                        nn.ReLU(),
        #                        nn.Dropout(0.5),
        #                        nn.Linear(self.hidden_dim, self.out_dim)).to(device)

        #self.Q_LSTM = Question_LSTM(input_size=2048).to(device)
        self.Q_LSTM = nn.LSTM(
            input_size=300,
            hidden_size=512,
            num_layers=1,  
            batch_first=True,
            bidirectional=True)
        self.A_LSTM = nn.LSTM(
            input_size=300,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.label_LSTM = nn.LSTM(
            input_size=300,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=False)
        #self.A_LSTM = Question_LSTM().to(device)
        #self.A_LSTM = self.Q_LSTM
        #self.FC = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim),
        #                        nn.ReLU(),
        #                        nn.Dropout(0.5),
        #                        nn.Linear(self.hidden_dim, self.out_dim)).cuda()
        self.FC_Answer = nn.Sequential(nn.Linear(1024, 1)).cuda()
        self.attn_linear = nn.Sequential(nn.Linear(1024, 1024), nn.Tanh()).to(device)
        self.temp_attn = nn.Sequential(nn.Linear(2048, 512), nn.Tanh(), nn.Linear(512, 1)).cuda()
        self.seq_pred = nn.Sequential(nn.Linear(4096, 256), nn.Tanh(), nn.Linear(256, 2)).cuda()
        #self.FC_Answer2 = nn.Sequential(nn.Linear(512, 144)).cuda()
        #self.FC_Output = nn.Sequential(nn.Linear(1024, 300), nn.Tanh(), nn.Dropout(0.1)).cuda()
        #self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.lang_linear = nn.Linear(768, 512)
        
        self.lang_layer = torch.nn.Sequential(
            self.lang_linear,
            nn.Tanh(),
            nn.Dropout(0.3))
            #nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU())
        # self.q_linear = nn.Sequential(nn.Linear(768, 512))
        #self.a_lang_layer = torch.nn.Sequential(
        #    self.lang_linear,
        #    self.a_norm,
        #    nn.ReLU(),
        #    nn.Dropout(0.3))
        #self.seg_lang_layer = torch.nn.Sequential(
        #    self.lang_linear,
        #    self.seg_norm,
        #    nn.ReLU(),
        #    nn.Dropout(0.3))
            
            
        self.embeds_QA = nn.Embedding(int(self.vocab_size), 300).to(device)
        #d_model = 512  # Original res feature is 512d

        self.q_layer1 = nn.Sequential(nn.Linear(1024, 512), nn.Tanh()).to(device)
        self.v_layer1 = nn.Sequential(nn.Linear(3072, 1024), nn.Tanh()).to(device)
        self.fc_max = nn.Sequential(nn.Linear(512, 1)).to(device)
        
        
        output_size = 512
        if self.model != 'QA':
            self.vq_fusion = nn.Sequential(nn.Linear(fc_n_feature-512, output_size), # hard coded 512...
                                           nn.Tanh()).to(device)
            self.q_fusion = nn.Sequential(nn.Linear(512, output_size),
                                          nn.Tanh()).to(device)
            self.va_fusion = nn.Sequential(nn.Linear(fc_n_feature-512, 512), nn.Tanh()).to(device)
    def forward(self, img_feat, q, choices, encoder=None):
        pass

    # ===== Multiple heads =====
    def multiheads(self, VQ_feature, Q_feature, choices, choices_mask, num_seg):
        batch_size = Q_feature.size(0)
        n_choices = choices.size(1)
        if True:
            # permute the answer order during training (we keep it same for each batch for now)
            idx_permute = torch.randperm(n_choices)
            target = VQ_feature.new(batch_size, n_choices).fill_(0)
            target[:, (idx_permute == 0)] = 1
            #print(idx_permute)
            #print(target)
            ch_bert = []
            choices_permuted = choices[:, idx_permute]
            #choices_mask_permuted = choices_mask[:, idx_permute]
            #print(idx_permute)
            #print(choices_permuted[:, 0, :])
            #print(choices_mask[:, 0, :])
            
            #for i in range(n_choices):
            #    ch_i_bert_seq_out, ch_bert_i_pool_out, ch_bert_i_all_encoder_layers = self.bert(choices_permuted[:, i, :], token_type_ids=None, attention_mask=choices_mask_permuted[:, i, :])
                #ch_bert_i_encoded = (ch_bert_i_all_encoder_layers[-1][:,0,:] + ch_bert_i_all_encoder_layers[-2][:,0,:]\
                #                   + ch_bert_i_all_encoder_layers[-3][:,0,:] + ch_bert_i_all_encoder_layers[-4][:,0,:])/4
            #    ch_bert_i_encoded = ch_bert_i_pool_out
            #    ch_bert_i_encoded = ch_bert_i_encoded.unsqueeze(1)#.detach()
            #    ch_bert.append(ch_bert_i_encoded)
            
            #ch_bert = torch.cat(ch_bert, dim=1)
            
            embeded_A = self.embeds_QA(choices[:, idx_permute].long().to(device))
            A_feature_list = []
            #hn = Q_feature.reshape(-1, 2, 512).permute(1, 0, 2).contiguous()
            #print(hn.shape)
            #print(n_choices, batch_size)
            for i in range(n_choices):
                A_feature_out, (A_feature_hn, A_feature_cn) = self.A_LSTM(embeded_A[:, i, :, :])#, (hn, torch.zeros((hn.shape)).cuda()))
                '''
                A_feature_batch = []
                for j in range(batch_size):
                    A_feature_j = A_feature_out[j, :, :].squeeze(0)[num_seg[j]-1, :].unsqueeze(0)
                    A_feature_batch.append(A_feature_j)
                '''
                #A_feature_out = A_feature_out.view((A_feature_out.size(0), A_feature_out.size(1), 2, 512))
                A_feature_batch_i = A_feature_out[:, -1, :]
                #A_feature_batch_i = self.q_layer1(A_feature_out[:, -1, :])
                #print(A_feature_batch_i.shape)
                A_feature_list.append(A_feature_batch_i.unsqueeze(1))
            A_feature = torch.cat(A_feature_list, dim=1)
            
            #A_feature_out, (A_feature_hn, A_feature_cn) = self.A_LSTM(embeded_A.view(-1, embeded_A.size(2), embeded_A.size(3)))
            #print(A_feature.shape)
            #A_feature_out = A_feature_out.view(embeded_A.size(0), 5, -1)
            
        else:
            # the correct answer is always at the begining for evaluation
            target = VQ_feature.new(batch_size, n_choices).fill_(0) # dummy
            idx_permute = torch.randperm(n_choices)
            target[:, (idx_permute == 0)] = 1
            #ch_bert = []
            ##choices_permuted = choices[:, idx_permute]
            #for i in range(n_choices):
            #    ch_bert_i, _ = self.bert(choices[:, i, :])
            #    ch_bert_i = torch.mean(ch_bert_i, 1).unsqueeze(1)
            #    ch_bert.append(ch_bert_i)
            #ch_bert = torch.cat(ch_bert, dim=1)
            #ch_bert_permuted = ch_bert[:, idx_permute]
            #embeded_A = self.embeds_QA(choices[:, idx_permute].long().to(device))
        #A_feature = self.A_LSTM(embeded_A.view(-1, embeded_A.size(2), embeded_A.size(3)), \
        #    ).squeeze().view(embeded_A.size(0), 5, -1)
        #A_feature = self.lang_layer(ch_bert)
        A_feature_list = []
        '''
        with torch.no_grad():
            for i in range(n_choices):
                A_feature_i = self.base_emb(choices_permuted[:, i, :], is_video=False).unsqueeze(1)
                A_feature_list.append(A_feature_i)
        A_feature = torch.cat(A_feature_list, dim=1)
        '''
        #A_feature = self.base_emb(choices_permuted, is_video=False)
        #A_feature = A_feature.detach()
        #A_feature = self.linear(A_feature)
        #if self.model == 'QA':
        #    out_feature = Q_feature.unsqueeze(1).expand_as(A_feature)
        #    #cat_feature = torch.cat([Q_feature.unsqueeze(1).expand_as(A_feature), A_feature], dim=-1)
        #else:
        #print(VQ_feature.shape)
        #print(A_feature.shape)
        #A_feature = self.FC_Answer(A_feature)
        VQ_feature = VQ_feature.unsqueeze(1).repeat((1, n_choices, 1))
        #print(vq_feature.shape)
        #v_feature = Q_feature.unsqueeze(1).repeat((1, n_choices, 1))
        #print(v_feature.shape)
        A_feature_normalized = nn.functional.normalize(A_feature, p=2, dim=-1)
        #v_feature_normalized = nn.functional.normalize(v_feature, p=2, dim=-1)
        vq_feature_normalized = nn.functional.normalize(VQ_feature, p=2, dim=-1)
        #va_feature = self.va_fusion(torch.cat([v_feature_normalized, A_feature_normalized], dim=-1))
        #print(va_feature.shape)
        #va_feature_normalized = nn.functional.normalize(va_feature, p=2, dim=-1)
        out_feature = A_feature_normalized*vq_feature_normalized
        #print(out_feature.shape)
        #print('-'*10)
        #print(ch_bert[0][0][:5])
        #print(ch_bert[0][1][:5])
        #print(ch_bert[0][2][:5])
        #print(ch_bert[0][3][:5])
        #print(ch_bert[0][4][:5])
        outs = self.FC_Answer(out_feature).squeeze(2)
        
        #print(out_feature.shape, A_feature.shape)
        #out_feature = torch.rand(out_feature.shape).cuda()
        #cat_feature = torch.cat([out_feature, A_feature], dim=-1)
        #print(cat_feature[0, 0, -5:])
        #print(cat_feature[0, 1, -5:])
        #out_feature = self.FC_Output(out_feature)
        #A_feature = self.FC_Answer(A_feature)
        #A_feature = torch.rand(A_feature.shape).cuda()
        #A_feature_normalized = nn.functional.normalize(A_feature, p=2, dim=-1)
        #out_feature_normalized = nn.functional.normalize(out_feature, p=2, dim=-1)
        #cat_feature = torch.cat([out_feature_normalized, A_feature_normalized], dim=-1)  
        #outs = self.vq_fusion(cat_feature).squeeze()
        #outs = nn.Softmax(dim=1)(outs)
        #cossim = torch.sum(A_feature_normalized * out_feature_normalized, 2)
        #print(outs[0])
        #outs = nn.Softmax(dim=1)(outs)
        #print(outs[0])
        #print(cossim[0])
        #print(outs)
        return outs, torch.argmax(target, 1)
