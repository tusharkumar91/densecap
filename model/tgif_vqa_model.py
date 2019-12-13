import torch
import random
import pickle
import time
import numpy as np
from torch import nn
from model.model import Net
from model.TGIFProtocolNet import ProtocolNet
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

class TGIFTransformerBaseline(ProtocolNet):
    def __init__(self, vocab_size, mode='Trans', d_model=1024, in_emb_dropout=0.1):
        self.name = mode
        self.vocab_size = vocab_size
        fc_n_feature = 512
        super(TGIFTransformerBaseline, self).__init__(1536 + 512, mode)
        #self.frame_emb = FrameEncoder(1024, in_emb_dropout, 2048, 2, 8, 0.2)
        self.vid_lstm = nn.LSTM(
            input_size=3072,
            hidden_size=1024,
            batch_first=True,
            
        ).cuda()
        self.rgb_emb = nn.Linear(2048, d_model // 2)
        self.flow_emb = nn.Linear(1024, d_model // 2)
        self.emb_out = nn.Sequential(
             #nn.Linear(d_model, d_model),
             # nn.BatchNorm1d(h_dim),
             nn.Dropout(0.1),
             nn.Tanh()
        )
        #
        # self.vis_emb = Transformer(d_model, 0, 0,
        #                            d_hidden=2048,
        #                            n_layers=2,
        #                            n_heads=8,
        #                            drop_ratio=0.2)


    def forward(self, vid_feat, seg, seg_mask, q, q_mask, choices, choices_mask, num_seg):
        # print("forward called")
        tic = time.time()
        tok = time.time()
        # print(question)
        #print(q)
        #print(q_mask)
        #q_seq_out, q_pool_out, q_all_encoder_layers = self.bert(q, token_type_ids=None, attention_mask=q_mask)
        #q_bert = (q_all_encoder_layers[-1][:,0,:] + q_all_encoder_layers[-2][:,0,:]\
        #          + q_all_encoder_layers[-3][:,0,:] + q_all_encoder_layers[-4][:,0,:])/4
        #Q_feature = q_pool_out
        #q_bert = q_bert.detach()
        #Q_feature = self.lang_layer(q_bert)
        #print(len(q_bert))
        #print(out_bert[0].shape)
        #q_feat = torch.mean(q_bert, 1)
        #Q_feature = self.q_linear(q_feat)
        #exit(0)
        embeded_Q = self.embeds_QA(q.long().cuda())
        seg_feature_out, (seg_feature_hn, seg_feature_cn) = self.Q_LSTM(embeded_Q)

        #embeded_label = self.embeds_QA(seg.long().cuda())
        #print(seg_feature_out.shape)
        #print(Q_feature)
        seg_text_cat_feat = []
        seg_bert = []
        batch_size = q.size(0)
        #print(num_seg)
        text_seg_mask = torch.ones((batch_size, 17, 768)).cuda()
        #for i in range(batch_size):
        #    text_seg_mask[i, num_seg[i]:, :] = 0
        
        '''
        #seg_vid_feat = []
        for i in range(17):
            x_rgb, x_flow = torch.split(vid_feat[:, i, :].squeeze(), 2048, 1)
            x_rgb = self.rgb_emb(x_rgb.contiguous())
            x_flow = self.flow_emb(x_flow.contiguous())
            x = torch.cat((x_rgb, x_flow), 1)
            x = self.emb_out(x)
            seg_vid_feat.append(x.unsqueeze(0))
        seg_vid_feat = torch.cat(seg_vid_feat, dim=0)
        
        seg_feature_out, (seg_feature_hn, seg_feature_cn) = self.vid_lstm(seg)
        V_feature_seg = seg_feature_out
        V_feature_list = []
        for i in range(batch_size):
            #print(V_feature_seg[i, :, :].shape, num_seg[i])
            # take nth hidden feature
            V_feature_i = V_feature_seg[i, :, :].squeeze(0)[num_seg[i]-1, :].unsqueeze(0)
            #V_feature_i = V_feature_seg[:, i, :].squeeze(1)[:num_seg[i], :].permute(1, 0).unsqueeze(0)
            #V_feature_i = nn.MaxPool1d(kernel_size=V_feature_i.size(2))(V_feature_i).squeeze(2)
            V_feature_list.append(V_feature_i)
            #print(V_feature_i.shape)
        V_feature = torch.cat(V_feature_list, dim=0)

        #text_seg_mask = torch.ones((batch_size, 17, 1024)).cuda()
        #for i in range(batch_size):
        #    text_seg_mask[i, seg_mask[i]:, :] = 0        
        with torch.no_grad():            
            for i in range(17):
                seg_text_i_feat = self.base_emb(vid_feat[:, i, :], is_video=True).unsqueeze(1)
                seg_text_cat_feat.append(seg_text_i_feat)
                #print(seg_text_i_feat)
            seg_text_cat_feat = torch.cat(seg_text_cat_feat, dim=1)
            #print(seg_text_cat_feat.shape)
            Q_feature = self.base_emb(q, is_video=False)
            #Q_feature = Q_feature.detach()
            #print(Q_feature.shape)
        #print(seg[:, 0, :])
        #print(seg[:, 1, :])
        #print(seg_mask[:, 0, :])
        #print(seg_mask[:, 1, :])
        '''
        '''
        for i in range(17):
            seg_i_seq_out, seg_i_pool_out, seg_i_all_encoder_layers = self.bert(seg[:, i, :], token_type_ids=None, attention_mask=seg_mask[:, i, :])
            #seg_i_bert = (seg_i_all_encoder_layers[-1][:,0,:] + seg_i_all_encoder_layers[-2][:,0,:]\
            #            + seg_i_all_encoder_layers[-3][:,0,:] + seg_i_all_encoder_layers[-4][:,0,:])/4
            seg_i_bert = seg_i_pool_out
            seg_i_bert = seg_i_bert.unsqueeze(1)
            seg_bert.append(seg_i_bert)
        seg_bert = torch.cat(seg_bert, dim=1)
        
        seg_feature_out, (seg_feature_hn, seg_feature_cn) = self.vid_lstm(seg_bert)
        '''
        Q_feature_seg = seg_feature_out
        Q_feature_list = []
        label_feature_list = []
        #label_feat = torch.zeros((batch_size, 17, 1024)).cuda()
        for i in range(batch_size):
            #print(V_feature_seg[i, :, :].shape, num_seg[i])
            # take nth hidden feature
            Q_feature_i = Q_feature_seg[i, :, :].squeeze(0)[num_seg[i], :].unsqueeze(0)
            #label_feature_i = torch.zeros((17, 1024)).cuda()
            #for j in range(seg_mask[i]):
            #    label_feature_j_out, (label_feature_j_hn, label_feature_j_cn) = self.Q_LSTM(embeded_labels[i, j].unsqueeze(0))
            #    label_feature_i[j] = label_feature_j_hn.reshape(-1).squeeze()
            #label_feat[i] = label_feature_i
            # mean of all features till n
            #V_feature_i = V_feature_seg[i, :, :].squeeze(0)[:num_seg[i], :].mean(dim=0).unsqueeze(0)
            #print(V_feature_i.shape)
            #V_feature_i = V_feature_seg[i, :, :].squeeze(0)[:num_seg[i], :].permute(1, 0).unsqueeze(0)
            #V_feature_i = nn.MaxPool1d(kernel_size=V_feature_i.size(2))(V_feature_i).squeeze(2)
            Q_feature_list.append(Q_feature_i)
            #print(V_feature_i.shape)
        Q_feature = torch.cat(Q_feature_list, dim=0)
        #Q_feature_max = Q_feature.unsqueeze(1).repeat((1, 17, 1))
        #label_feature_max = Q_feature_max + label_feat
        #label_feature_max = self.fc_max(label_feature_max).squeeze(-1)
        #print(label_feature_max.shape)
        #print('label max fc out', label_feature_max[0])
        #vid_max_seg = []
        #for i in range(batch_sizçe):
        #    #print(label_feature_max[i, :seg_mask[i]])
        #    vid_max_seg_i = torch.argmax(label_feature_max[i, :seg_mask[i]])
        #    #print(vid_max_seg_i)
        #    vid_max_seg.append(vid_max_seg_i)
        #print('input video feature', seg.shape, seg[0][0][:10])
        #a = pickle.load(open('input_seqpred_video.pkl', 'rb'))
        #b = pickle.load(open('output_seqpred_video.pkl', 'rb'))
        #a.append(seg[0][0].cpu().detach().numpy())
        #print('rgb flow')
        #pickle.dump(a, open('input_seqpred_video.pkl', 'wb'))
        #x_rgb, x_flow = torch.split(seg, 2048, 2)
        #vid = self.v_layer1(seg)
        #x_rgb = self.rgb_emb(x_rgb.contiguous())
        #x_flow = self.flow_emb(x_flow.contiguous())
        #x = torch.cat((x_rgb, x_flow), 2)
        #print('input emb feature', x.shape, x[0][0][-10:])
        #x = self.emb_out(x)
        #x = torch.rand((x.shape)).cuda()
        #Q_fusion_feature = Q_feature.unsqueeze(1).repeat((1, x.shape[1], 1))
        #VQ_feature = torch.cat((x, Q_fusion_feature), dim=-1)
        #V_feature, all_emb, _ = self.frame_emb(x, None)
        #V_feature = nn.MaxPool1d(kernel_size=V_feature.shape[1])(V_feature.view(V_feature.size(0), 1024, -1)).squeeze(2)
        #vid_feat = torch.rand((vid_feat.shape)).cuda()
        #Q_feature = self.q_layer1(Q_feature)
        #label_feat_kb = self.q_layer1(label_feat)
        kb = torch.zeros((batch_size, 1024)).cuda()
        #kb = Q_feature.clone()
        hn = Q_feature.clone()
        #hn = torch.zeros((batch_size, 1024)).cuda() 
        #print('in lstm')
        for _ in range(1):
            vid_feature_list = []
            vid_feature_out, (vid_feature_hn, vid_feature_cn) = self.vid_lstm(seg, (hn.unsqueeze(0), torch.zeros((hn.unsqueeze(0).shape)).cuda()))
            for i in range(batch_size):
                vid_feature_i = vid_feature_out[i, :, :].squeeze(0)[seg_mask[i]-1, :].unsqueeze(0)
                #tgif attention
                Q_attn_feature = Q_feature[i, :].unsqueeze(0).repeat((seg_mask[i], 1))
                input_attn_feature = torch.cat((Q_attn_feature, vid_feature_out[i, :, :].squeeze(0)[:seg_mask[i], :]), dim=-1)
                seq_attn = self.temp_attn(input_attn_feature)
                seq_attn = seq_attn.reshape((1, -1))
                v_attn_feature = torch.matmul(seq_attn, vid_feature_out[i, :, :].squeeze(0)[:seg_mask[i], :])
                v_attn_feature = self.attn_linear(v_attn_feature)
                vid_feature_list.append(v_attn_feature)
                #kb[i] += vid_feature_i.squeeze()
            #hn = kb.clone()
        #print('out lstm')
        #print(vid_feature_out.shape)
        #for i in range(batch_size):
        #    #vid_feature_i = label_feat[i, num_seg[i], :].unsqueeze(0)
        #    vid_feature_i = vid_feature_out[i, :, :].squeeze(0)[seg_mask[i]-1, :].unsqueeze(0)
        #    vid_feature_list.append(vid_feature_i)
        V_feature = torch.cat(vid_feature_list, dim=0)
        #V_feature = vid_feature_out
        #V_feature = torch.rand((V_feature.shape)).cuda()
        #b.append(V_feature[0].cpu().detach().numpy())
        #pickle.dump(b, open('output_seqpred_video.pkl', 'wb'))
        #V_feature = torch.rand((batch_size, 768)).cuda()
        #seg_bert = seg_bert.detach()
        
        #seg_feature = self.lang_layer(seg_bert)
        #print(seg_feature.shape)
        #print(Q_feature.shape)
        #seg_text_cat_feat = self.linear(seg_text_cat_feat)
        #print(seg_text_cat_feat)
        #Q_feature = self.linear(Q_feature)
        #Q_cat_feature = Q_feature.unsqueeze(1).repeat((1, 17, 1))
        #print(Q_cat_feature.shape)
        #seg_q_cat_feature = torch.cat((seg_text_cat_feat, Q_cat_feature), dim=-1)
        #print(seg_q_cat_feature.shape)
        #seg_feature_out, (seg_feature_hn, seg_feature_cn) = self.Q_LSTM(seg_text_cat_feat)
        
        #seg_text_cat_feat = seg_text_cat_feat.detach()
        tic = time.time()
        #seg_feature_out, (seg_feature_hn, seg_feature_cn) = self.Q_LSTM(seg_text_cat_feat)
        #V_feature = self.linear(seg_text_cat_feat)
        #V_feature, all_emb, _ = self.frame_emb(seg_text_cat_feat, text_seg_mask)
        #Q_feature = self.linear(Q_feature)
        #print('v feature after transformer', V_feature.shape)
        #V_feature = nn.MaxPool1d(kernel_size=V_feature.shape[1])(V_feature.view(V_feature.size(0), 1024, -1)).squeeze(2)
        #V_feature = self.linear_v(V_feature)
        #V_feature = self.linear(V_feature)
        #print(V_feature.shape)
        #print(Q_feature.shape)
        #tok = time.time()
        #seg_feature_hn = seg_feature_hn.permute((1, 0, 2)).view(Q_feature.size(0), -1)
        #V_feature = seg_feature_out
        #V_feature = nn.MaxPool1d(kernel_size=V_feature.shape[1])(V_feature.contiguous().view(V_feature.size(0), 1024, -1)).squeeze(2)
        #V_feature = torch.rand((batch_size, 1024)).cuda()
        
        #VQ_feature = torch.cat([V_feature, Q_feature], dim=1)
        #VQ_feature = torch.cat([nn.functional.layer_norm(V_feature, [V_feature.shape[-1]]),
        #                        nn.functional.layer_norm(Q_feature, [Q_feature.shape[-1]])], dim=1)
        #print(VQ_feature.shape)
        #Q_feature = self.q_layer1(Q_feature)
        #V_feature = torch.rand((V_feature.shape)).cuda()
        #print(torch.mean(V_feature))
        #print('q feature', Q_feature[0][:10])
        #print('output lstm', V_feature.shape, V_feature[0][:10])
        #V_feature_out = self.v_layer1(V_feature)
        #V_feature_out = kb
        
        #cls_input = torch.zeros((3*batch_size, 1536)).cuda()
        #cls_output = torch.zeros((3*batch_size)).long().cuda()

        cls_input = []
        cls_output = []
        #vid_feat = self.q_layer1(x)
        #vid_feat = x_rgb
        #label_feat_kb = self.q_layer1(label_feat)
        #label_feat_kb = label_feat
        input_idx = 0
        '''
        seg_frame_features = []
        for i in range(batch_size):
            seg_frame_feat_i = []
            for num in np.arange(0, seg_mask[i], 50):
                start = num
                end = num+50
                if end > seg_mask[i]:
                    end = seg_mask[i]
                seg_frame_feat_i.append(torch.mean(vid[i, start:end, :], dim = 0))
                #print(start, end)
            seg_frame_features.append(seg_frame_feat_i)
        
        for i in range(batch_size):
            pairs = []
            #if len(seg_frame_features[i]) < 2:
            #    continue
            #for _ in range(1):
            #    first = random.choice(range(len(seg_frame_features[i])-1))
            #    second = random.choice(range(first+1, len(seg_frame_features[i])))
            #    #second = first+1
            #    pairs.append((first, second))
            #    #print(first, second)
            pairs = [(k, random.choice(range(k+1, seg_mask[i]-1))) for k in range(seg_mask[i]-2)]
            #print(pairs)
            random.shuffle(pairs)
            #print(pairs)
            pairs = pairs
            #before 0 after 1
            for pair in pairs:
                p = random.random()
                #print(kb[i].shape)
                if p < 0.5:
                    x1, y1 = pair
                    label = 0
                    #print(seg_frame_features[i][x1].shape)
                    #cls_input[input_idx] = torch.cat([kb[i], label_feat_kb[i, x], label_feat_kb[i, y]])
                    
                    cls_input.append(torch.cat([nn.functional.normalize(seg[i,x1,:], p=2, dim=-1), nn.functional.normalize(seg[i, y1, :], p=2, dim=-1)]).unsqueeze(0))
                else:
                    y1, x1 = pair
                    #print(seg_frame_features[i][x1].shape)
                    label = 1
                    #cls_input[input_idx] = torch.cat([kb[i], label_feat_kb[i, y], label_feat_kb[i, x]])
                    cls_input.append(torch.cat([nn.functional.normalize(seg[i, y1, :], p=2, dim=-1), nn.functional.normalize(seg[i, x1, :], p=2, dim=-1)]).unsqueeze(0))
                #cls_output[input_idx] = label
                cls_output.append(label)
                input_idx += 1
        cls_input = torch.cat(cls_input, dim=0).cuda()
        cls_output = torch.LongTensor(cls_output).cuda()
        #print(cls_input.shape, cls_output.shape)
        cls_pred = self.seq_pred(cls_input)        
        print(torch.argmax(cls_pred, dim=1))
        '''
        #print('output vid layer', V_feature_out.shape, V_feature_out[0][:10])
        #params = torch.cat(list(self.v_layer1[0].parameters()), dim=0)
        #print(torch.min(torch.mean(params, dim=0)))
        #print(torch.max(torch.mean(params, dim=0)))
        Q_feature_normalized = nn.functional.normalize(Q_feature, p=2, dim=-1)
        Q_feature_l1_normalized = nn.functional.normalize(Q_feature, p=2, dim=-1)
        V_feature_l1_normalized = nn.functional.normalize(V_feature, p=2, dim=-1)
        VQ_feature_normalized = nn.functional.normalize(V_feature_l1_normalized, p=2, dim=-1)
        #VQ_feature_normalized = torch.zeros((Q_feature_normalized.shape)).cuda()
        Q_feature_out = Q_feature_normalized+VQ_feature_normalized
        #print('q norm', Q_feature_l1_normalized[0][:10])
        #print('max q', torch.max(Q_feature_l1_normalized[0]))
        #print('output norm', V_feature_l1_normalized.shape, V_feature_l1_normalized[0][:10])
        #print('max ', torch.max(V_feature_l1_normalized[0]))
        #print(torch.mean(torch.norm(Q_feature_out - Q_feature_normalized, p=2, dim=1)), torch.mean(torch.norm(Q_feature_out, p=2, dim=-1)))
        #VQ_feature = self.vq_fusion(VQ_feature)
        #outs = self.multiheads(VQ_feature, Q_feature, choices)
        outs = self.multiheads(Q_feature_out, Q_feature, choices, choices_mask, num_seg)
        #outs = self.FC_Answer(Q_feature_out)
        #outs = self.FC_Answer2(outs)
        #print(outs.shape)
        #print(cls_pred[0], cls_output[0])
        return outs#, (cls_pred, cls_output)
