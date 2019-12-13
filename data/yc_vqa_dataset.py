import torch
import torch.utils.data
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
import math
import torch.nn.functional as F
import re
from transformers import *
from gensim.models.keyedvectors import KeyedVectors


class YouCookQADataset(torch.utils.data.Dataset):
    def __init__(self, image_path, split='training', slide_window_size=480, we=None):
        self.split = split
        self.slide_window_size = slide_window_size
        self.vocab = {}
        self.qa_ans_vid_data = []
        self.image_path = image_path
        self.split_path = os.path.join(image_path, split)
        self.video_id_idx_map = {}
        self.video_segment_map = {}
        self.captions = {}
        self.video_features = {}
        self.we = we
        self.vid_seg_data = np.load(os.path.join(self.image_path,"mat_frame.pkl.npy"))
        self.verbs_vid = {}
        self.nouns_vid = {}
        self.get_segment_data()
        self.vid_rgb_feat = {}
        self.print_cls = False
        #self.get_autogen_data()
        #self.initialize_vocab()
        self.initialize_mask_q()
        #self.initialize_data()
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def initialize_mask_q(self):
        self.nouns = pickle.load(open(os.path.join(self.image_path, 'nouns.pkl'), 'rb'))
        self.verbs = pickle.load(open(os.path.join(self.image_path, 'verbs.pkl'), 'rb'))
        with open(os.path.join(self.image_path, 'vocab_noun_and_seg_autogen_q_opt_v1.json')) as f:
            vocab = json.load(f)
        self.vocab = vocab
        with open(os.path.join(self.image_path, 'noun_and_seg_autogen_q_opt_v1.json')) as f:
            questions =json.load(f)
        self.verbs_vid = pickle.load(open(os.path.join(self.image_path, 'verb_vid_v1.pkl'), 'rb'))
        self.nouns_vid = pickle.load(open(os.path.join(self.image_path, 'noun_vid_v1.pkl'), 'rb'))

        #if self.split == 'training':
        howto100data_train = pickle.load(open(os.path.join(self.image_path, 'youcook_train.pkl'), 'rb'))
        #else:
        howto100data_val = pickle.load(open(os.path.join(self.image_path, 'youcook_val.pkl'), 'rb'))
        howto100data_test = pickle.load(open(os.path.join(self.image_path, 'youcook_test.pkl'), 'rb'))
        howto100data = []
        for idx in range(len(howto100data_train)):
            howto100data.append(howto100data_train[idx])
        for idx in range(len(howto100data_val)):
            howto100data.append(howto100data_val[idx])
        for idx in range(len(howto100data_test)):
            howto100data.append(howto100data_test[idx])
        print('Working on extracting caption 2d and 3d features for {} segments'.format(len(howto100data)))
        for idx in range(len(howto100data)):
            feat_2d = howto100data[idx]['2d']
            feat_3d = howto100data[idx]['3d']
            vid_id = howto100data[idx]['id']
            caption = howto100data[idx]['caption']
            self.captions[vid_id] = caption
            self.video_features[vid_id] = {'2d' : feat_2d, '3d' : feat_3d}
        
            
        answer_classes = {}
        cls_count = -1
        qs = []
        vids = []
        for qa_idx, qa in enumerate(tqdm(questions)):
            #if qa["answer"] not in answer_classes:
            #    cls_count += 1
            #    answer_classes[qa["answer"]] = cls_count
            if qa["subset"] == self.split:
                vid = qa["video_id"]
                howtovid_idx = vid + '_'  + str(0)
                if howtovid_idx not in self.captions:
                    continue
                vids.append(vid)
                q_tokens = qa['question'].split(' ')
                a_tokens = qa['answer'].split(' ')
                opts = []
                for opt in qa['alternatives']:
                    opt_tokens = opt.split(' ')
                    opts.append(opt_tokens)
                q = np.zeros((30), dtype=np.int)
                ch = np.zeros((5, 30), dtype=np.int)
                mask_count = 0
                if len(a_tokens) == 1:
                    q_idx = 0
                    mask_idx = None
                    for idx, token in enumerate(q_tokens):
                        if token in self.vocab:
                            q[q_idx] = self.vocab[token]
                            if token == '[MASK]':
                                mask_count += 1
                                mask_idx = q_idx
                            q_idx += 1
                    for idx in range(5):
                        ch[idx, :] = q
                    ch[0, mask_idx] = self.vocab[a_tokens[0]]
                    for idx in range(4):
                        ch[idx+1, mask_idx] = self.vocab[opts[idx][0]]
                else:
                    q_idx = 0
                    for idx, token in enumerate(q_tokens):
                        if token in self.vocab:
                            q[q_idx] = self.vocab[token]
                            if token == '[MASK]':
                                mask_count += 1
                            q_idx += 1
                    a_idx = 0
                    for idx, token in enumerate(a_tokens):
                        if token in self.vocab:
                            ch[0, a_idx] = self.vocab[token]
                            a_idx += 1
                    ch_idx = 1
                    for opt in opts:
                        a_idx = 0
                        for idx, token in enumerate(opt):
                            if token in self.vocab:
                                ch[ch_idx, a_idx] = self.vocab[token]
                                a_idx += 1
                        ch_idx += 1
                if mask_count == 1:
                    qs.append((q, ch))
                '''
                if vid not in self.vid_rgb_feat:
                    video_prefix = os.path.join(self.split_path, vid)
                    #print(video_prefix)
                    resnet_feat = torch.from_numpy(
                            np.load(video_prefix + '_resnet.npy')).float()
                    bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
                    img_feat = torch.FloatTensor(np.zeros((480,
                                                           resnet_feat.size(1) + bn_feat.size(1))))
                    total_frame = bn_feat.size(0)
                    torch.cat((resnet_feat, bn_feat), dim=1,
                                        out=img_feat[:min(total_frame, 480)])
                    self.vid_rgb_feat[vid] = (img_feat, min(total_frame, 480))
                '''
                #a = answer_classes[qa["answer"]]
                #for q_token in q_tokens:
                #    if q_token in vocab:
                #        q.append(vocab[q_token.strip()])
                #    else:
                #        q.append(vocab['[UNK]'])
        self.qs = qs
        self.vocab = vocab
        self.vids = vids
        self.answer_cls = answer_classes
        
    def get_autogen_data(self):
        with open(os.path.join(self.image_path, "recipe1m_autogen_100K.json")) as f:
            recipe1m_qas = json.load(f)
        with open(os.path.join(self.image_path, "recipe1m_segment_100K.json")) as f:
            recipe1m_segments = json.load(f)
        for qa_idx, qa_pairs in enumerate(recipe1m_qas):
            if qa_idx % 1000 == 0:
                print('Done {}/{}'.format(qa_idx+1, len(recipe1m_qas)))
            recipe_segments = recipe1m_segments[qa_idx]
            segments_data = []
            for segment in recipe_segments:
                segment_data = []
                segment_tokens = word_tokenize(segment)
                for i, token in enumerate(segment_tokens):
                    segment_data.append(token)
                segments_data.append(segment_data)
            for qa in qa_pairs:
                if qa["split"] == self.split:
                    vid = qa["video_id"]
                    q_bert = []
                    q_tokens = word_tokenize(qa["question"])
                    for i, token in enumerate(q_tokens):
                        q_bert.append(token)
                    answer_bert = []
                    answer_tokens = word_tokenize(qa["answer"])
                    for i, token in enumerate(answer_tokens):
                        answer_bert.append(token)
                    choices_bert = []
                    choices_bert.append(answer_bert)
                    for opt_idx, opt in enumerate(qa["alternatives"]):
                        opt_tokens = word_tokenize(opt)
                        choice_bert = []
                        for i, token in enumerate(opt_tokens):
                            choice_bert.append(token)
                        choices_bert.append(choice_bert)
                    self.qa_ans_vid_data.append({
                        "question_bert" : q_bert,
                        "choices_bert" : choices_bert,
                        "segments" : segments_data,
                        "vid" : vid})
            #if qa_idx > 20:
            #    break
                            
                    
    def get_segment_data(self):
        vid_id = np.load(os.path.join(self.image_path,"mat_id.pkl.npy"))
        for i in range(len(vid_id)):
            if vid_id[i] not in self.video_id_idx_map:
                self.video_id_idx_map[vid_id[i]] = i
        
        
    def initialize_data(self):
        self.verbs_vid = pickle.load(open(os.path.join(self.image_path, 'verb_vid.pkl'), 'rb'))
        self.nouns_vid = pickle.load(open(os.path.join(self.image_path, 'noun_vid.pkl'), 'rb'))
        if self.split == 'training':
            howto100data = pickle.load(open(os.path.join(self.image_path, 'youcook_train.pkl'), 'rb'))
        else:
            howto100data = pickle.load(open(os.path.join(self.image_path, 'youcook_val.pkl'), 'rb'))
        print('Working on extracting caption 2d and 3d features for {} segments'.format(len(howto100data)))
        for idx in range(len(howto100data)):
            feat_2d = howto100data[idx]['2d']
            feat_3d = howto100data[idx]['3d']
            vid_id = howto100data[idx]['id']
            caption = howto100data[idx]['caption']
            self.captions[vid_id] = caption
            self.video_features[vid_id] = {'2d' : feat_2d, '3d' : feat_3d}
        
        #with open(os.path.join(self.image_path, "final_db_3K_combined_without_yesno.json")) as f:
        with open(os.path.join(self.image_path, "final_db_3K_auto_gen.json")) as f:
            qa_data = json.load(f)
        with open(os.path.join(self.image_path, "video_segments_map.json")) as f:
            segment_data = json.load(f) 
        for qa in qa_data["QAPairs"]:
            if qa["split"] == self.split:
                vid = qa["video_id"]
                howtovid_idx = vid + '_'  + str(0)
                if howtovid_idx not in self.captions:
                    continue
                q = np.zeros((36))
                q_tokens = word_tokenize(qa["question"])
                idx = -1
                q_bert = []
                if vid not in self.video_segment_map:
                    segments = segment_data[vid]
                    segment_tokenized = []
                    for seg_idx, segment in enumerate(segments):
                        seg_tokens = []
                        for token in word_tokenize(segment):
                            seg_tokens.append(token)
                        segment_tokenized.append(seg_tokens)
                    self.video_segment_map[vid] = segment_tokenized
                
                for i, token in enumerate(q_tokens):
                    q_bert.append(token)
                    #q[idx] = self.vocab[token]
                    idx = idx - 1
                pad_num = 36 - len(q_tokens)
                
                choices_bert = []
                answer_bert = []
                answers = np.zeros((5, 36))
                answer_tokens = word_tokenize(qa["answer"])
                if len(answer_tokens) < 2:
                    continue
                idx = -1
                for i, token in enumerate(answer_tokens):
                    answer_bert.append(token)
                    #answers[0][idx] = self.vocab[token]
                    idx = idx - 1
                choices_bert.append(answer_bert)
                for opt_idx, opt in enumerate(qa["alternatives"]):
                    idx = -1
                    opt_tokens = word_tokenize(opt)
                    choice_bert = []
                    for i, token in enumerate(opt_tokens):
                        choice_bert.append(token)
                        #answers[opt_idx + 1][idx] = self.vocab[token]
                        idx = idx - 1
                    choices_bert.append(choice_bert)
                self.qa_ans_vid_data.append({
                    "question" : q,
                    "choices" : answers,
                    "ans_idx" : 0,
                    "question_bert" : q_bert,
                    "choices_bert" : choices_bert,
                    "vid" : vid
                })
        print('Total {} qa pairs in {}'.format(len(self.qa_ans_vid_data), self.split))
            
            

    def initialize_vocab(self):
        #print('Loading word vectors')
        #self.we = KeyedVectors.load_word2vec_format(os.path.join(self.image_path, 'GoogleNews-vectors-negative300.bin'), binary=True)
        #print('Done loading word vectors')
        dict_dir = os.path.join(self.image_path, 'vocab_3K_combined_without_yesno.json')
        with open(dict_dir) as f:
            word_to_idx = json.load(f)
        self.vocab = word_to_idx
        # for word in word_to_idx:
        #     idx_to_word[word_to_idx[word]] = word
        # self.old_idx_to_word = idx_to_word
        # print(len(self.old_idx_to_word))

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size], 0
        else:
            zero = np.zeros((size - len(tensor), 300), dtype=np.float32)
            zero_count = size - len(tensor)
            return np.concatenate((tensor, zero), axis=0), zero_count
    
    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        return torch.zeros(36, 300), 36
        words = [word for word in words if word in self.we.vocab]
        #print(words)
        if words:
            we, count = self._zero_pad_tensor(self.we[words], 36)
            return torch.from_numpy(we), count
        else:
            return torch.zeros(36, 300), 36

    def __getitem__(self, idx):
        q_np, ch_np = self.qs[idx]
        #print(q_np)
        mask_idx = np.argwhere(q_np == 1)[0][0]
        #print(mask_idx)
        #print(ch_np)
        '''
        q_np = np.zeros((30), dtype=np.int)
        ch_np = np.zeros((5, 30), dtype=np.int)
        sent_idx = -1
        for tok_idx, token in enumerate(q):
            if token in self.vocab:
                sent_idx += 1
                q_np[sent_idx] = self.vocab[token]
                if token == '[MASK]':
                    ch_np[0, sent_idx] = self.vocab[a[0]]
                    for ch_idx in range(1, 5):
                        ch_np[ch_idx, sent_idx] = self.vocab[opts[ch_idx-1][0]]
                else:
                    ch_np[:, sent_idx] = self.vocab[token]
            #else:
            #    q_np[tok_idx] = self.vocab['[UNK]']
            #    print(ch_np[0].shape, tok_idx)
            #    ch_np[:, tok_idx] = self.vocab['[UNK]']
        '''
        #print(q_np)
        #print(ch_np[0])
        #print(ch_np[1])
        #print(ch_np[2])
        #print(ch_np[3])
        #print(ch_np[4])
              

        vid = self.vids[idx]
        segment_dets = torch.zeros((17, 300))
        segment_labels = torch.zeros((17, 20))
        video_segment_features = torch.zeros((17, 4096))        
        num_seg = 0
        
        for seg_idx in range(17):
            vid_key = vid + '_' + str(seg_idx)
            if vid_key not in self.captions:
                break
            else:
                nouns_dets = torch.from_numpy(self.nouns_vid[vid_key])
                verbs_dets = torch.from_numpy(self.verbs_vid[vid_key])
                
                topknounidx = np.argpartition(nouns_dets, -50)[-50:]

                nouns_pos_vals = (nouns_dets > 0.2).nonzero().reshape(-1)[:]
                verbs_pos_vals = (verbs_dets > 0.2).nonzero().reshape(-1)[:]
                noun_pos = []
                cnt = 0
                noun_pos_names = []
                for noun in topknounidx:
                    if self.nouns[noun] in self.vocab:
                        segment_labels[seg_idx][cnt] = self.vocab[self.nouns[noun]]
                        cnt += 1
                        noun_pos_names.append(self.nouns[noun])
                        if cnt > 19:
                            break
                '''
                verb_pos_names = []
                
                for verb in verbs_pos_vals:
                    if cnt < 0:
                        break
                    if self.verbs[verb] in self.vocab:
                        segment_labels[seg_idx][cnt] = self.vocab[self.verbs[verb]]
                        verb_pos_names.append(self.verbs[verb])
                        cnt -= 1
                
                verbs_dets = torch.from_numpy(self.verbs_vid[vid_key])
                '''
                segment_dets[seg_idx, :] = torch.cat([nouns_dets, verbs_dets])[:300]
                num_seg = seg_idx+1
                #print('nouns', noun_pos_names)
                #print(vid_key)
                #exit(0)
                #print('verbs', verb_pos_names)
                feat_2d = F.normalize(torch.from_numpy(self.video_features[vid_key]['2d']).float(), dim=0)
                feat_3d = F.normalize(torch.from_numpy(self.video_features[vid_key]['3d']).float(), dim=0)
                feat_vid = torch.cat((feat_2d, feat_3d))
                video_segment_features[seg_idx] = feat_vid
                #print('seg i ', self.captions[seg_i_key])
        #print(vid, num_seg)
        
        #print(vid)
        q_l = torch.zeros((1)).long()
        q_l[:] = 30
        ch_l = torch.zeros((5)).long()
        ch_l[:] = 30
        video_prefix = os.path.join(self.split_path, vid)
        #print(video_prefix)
        '''
        resnet_feat = torch.from_numpy(
            np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
        img_feat = torch.FloatTensor(np.zeros((self.slide_window_size,
                                               resnet_feat.size(1) + bn_feat.size(1))))
        total_frame = bn_feat.size(0)
        torch.cat((resnet_feat, bn_feat), dim=1,
                  out=img_feat[:min(total_frame, self.slide_window_size)])
        '''
        #self.vid_rgb_feat[vid][0], self.vid_rgb_feat[vid][1]

        return torch.zeros((1, 36)), segment_labels, num_seg, torch.from_numpy(q_np), q_l, torch.from_numpy(ch_np), ch_l, mask_idx
    
        '''
        q_text = ' '.join(self.qa_ans_vid_data[idx]["question_bert"])
        print('question : ', q_text)
        q_we, _ = self._words_to_we(self._tokenize_text(q_text))
        
        
        # get word vector of question
        ch_we_list = []
        for ch_idx, choice in enumerate(self.qa_ans_vid_data[idx]["choices_bert"]):
            ch_i_text = ' '.join(choice)
            ch_i_we, _ = self._words_to_we(self._tokenize_text(ch_i_text))
            print('choice i : ', ch_i_text)
            ch_i_we = ch_i_we.unsqueeze(0)
            ch_we_list.append(ch_i_we)
        ch_we = torch.cat(ch_we_list, dim=0)
        
        # get word vector of choices
        # get segment level features and concat and send it as an array
        '''
        q_text = ' '.join(self.qa_ans_vid_data[idx]["question_bert"])
        #print(q_text)
        q_bert = torch.tensor([self.tokenizer.encode(' '.join(self.qa_ans_vid_data[idx]["question_bert"]), add_special_tokens=True, max_length=40)]).squeeze()
        #print(q_bert)
        pad_q = torch.zeros((40), dtype=torch.long)
        pad_q[:q_bert.size(0)] = q_bert
        input_q_mask = torch.zeros((40),  dtype=torch.long)
        input_q_mask[:q_bert.size(0)] = 1
        ch = []
        input_ch_mask = []
        
        for ch_idx, choice in enumerate(self.qa_ans_vid_data[idx]["choices_bert"]):
            #print(' '.join(choice))
            ch_i_bert = torch.tensor([self.tokenizer.encode(' '.join(choice), add_special_tokens=True, max_length=40)])
            #print(ch_i_bert)
            pad_ch = torch.zeros((1, 40), dtype=torch.long)
            pad_ch[:, :ch_i_bert.size(1)] = ch_i_bert
            input_ch_i_mask = torch.zeros((1, 40),  dtype=torch.long)
            input_ch_i_mask[:, :ch_i_bert.size(1)] = 1
            ch.append(pad_ch)
            input_ch_mask.append(input_ch_i_mask)

        ch_bert = torch.cat(ch, dim=0)
        ch_mask = torch.cat(input_ch_mask, dim=0)
        #print(pad_q)
        #print(input_q_mask)
        #print(ch_bert)
        #print(ch_mask)
        
        qa_ans_vid_data_idx = self.qa_ans_vid_data[idx]
        #q = qa_ans_vid_data_idx["question"]
        #ch = qa_ans_vid_data_idx["choices"]
        #ans_idx = qa_ans_vid_data_idx["ans_idx"]
        vid = qa_ans_vid_data_idx["vid"]
        #data_idx = self.video_id_idx_map[vid]
        #vid_idx_seg_data = self.vid_seg_data[data_idx]
        
        segment_tokens = self.video_segment_map[vid]
        segment_data = torch.zeros((17, 40), dtype=torch.long)
        input_seg_mask = torch.zeros((17, 40),  dtype=torch.long)
        segment_dets = torch.zeros((17, 775))
        num_seg = 0
        vid
        for seg_idx, segment_token in enumerate(segment_tokens):
            #print(' '.join(segment_token))
            #seg_i_bert = torch.tensor([self.tokenizer.encode(' '.join(segment_token), add_special_tokens=True, max_length=40)]).squeeze()
            vid_key = vid + '_' + str(seg_idx)
            #nouns_dets = torch.from_numpy(self.nouns_vid[vid_key])
            #verbs_dets = torch.from_numpy(self.verbs_vid[vid_key])
            #segment_dets[seg_idx, :] = torch.cat([nouns_dets, verbs_dets])
            #print(seg_i_bert)
            #if seg_idx == 0:
            #    #print('segment token is this')
            #    #print(segment_token)
            #    #print('segment bert is this')
            #    #print(seg_i_bert)
            segment_data[seg_idx, :seg_i_bert.size(0)] = seg_i_bert
            input_seg_mask[seg_idx, :seg_i_bert.size(0)] = 1
            num_seg = seg_idx+1
        #print(num_seg)
        '''
        video_segment_features = torch.zeros((17, 4096))
        text_segment_features = torch.zeros((17, 36, 300))
        text_segment_mask = torch.zeros((17, 36))
        
        num_seg=0
        #for i, seg_tokens in enumerate(self.qa_ans_vid_data[idx]["segments"]):
        #    seg_i_key = vid + '_' + str(0)
        #    seg_i_text = ' '.join(seg_tokens)
        #    #print('seg : ', seg_i_text)
        #    text_segment_features[i], count = self._words_to_we(self._tokenize_text(seg_i_text))
        #    text_segment_mask[i] = torch.ones((36))
        #    if count > 0:
        #        text_segment_mask[i][-count:] = 0
        #num_seg = len(self.qa_ans_vid_data[idx]["segments"])
        #print(num_seg)
        
        for i in range(17):
            seg_i_key = vid + '_' + str(i)
            if seg_i_key not in self.video_features:
                num_seg = i
                break
            else:
                feat_2d = F.normalize(torch.from_numpy(self.video_features[seg_i_key]['2d']).float(), dim=0)
                feat_3d = F.normalize(torch.from_numpy(self.video_features[seg_i_key]['3d']).float(), dim=0)
                feat_vid = torch.cat((feat_2d, feat_3d))
                video_segment_features[i] = feat_vid
                #print('seg i ', self.captions[seg_i_key])
                text_segment_features[i], count = self._words_to_we(self._tokenize_text(self.captions[seg_i_key]))
                text_segment_mask[i] = torch.ones((36))
                if count > 0:
                    text_segment_mask[i][-count:] = 0
        
        #print(vid, num_seg)
        video_prefix = os.path.join(self.split_path, vid)
        resnet_feat = torch.from_numpy(
            np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
        img_feat = torch.FloatTensor(np.zeros((self.slide_window_size,
                                               resnet_feat.size(1) + bn_feat.size(1))))
        vid_seg_features = torch.zeros((17, 3072))
        #print(vid_idx_seg_data.shape)
        num_seg = 0
        for i in range(17):
            if int(vid_idx_seg_data[i][0]) == -1 or int(vid_idx_seg_data[i][1]) == -1:
                num_seg = i
                break
            else:
                vid_seg_features[i] = torch.mean(img_feat[int(vid_idx_seg_data[i][0]):int(vid_idx_seg_data[i][1])+1]).squeeze()

        total_frame = bn_feat.size(0)
        torch.cat((resnet_feat, bn_feat), dim=1,
                  out=img_feat[:min(total_frame, self.slide_window_size)])
        '''
        return torch.zeros((17, 36)), segment_data, input_seg_mask, pad_q, input_q_mask, ch_bert, ch_mask, num_seg
        #return video_segment_features, text_segment_features, torch.zeros((17, 36)), q_we, torch.zeros((17, 36)), ch_we, torch.zeros((17, 36)), num_seg

    def __len__(self):
        # if self.split == "validation":
        #    return len(q_ids)
        #    #return len(self.mat_id)
        # elif self.split == "training":
        #    return len(train_q_ids)
        # else:
        return len(self.qs)
        return len(self.qa_ans_vid_data)


if __name__ == "__main__":
    dataset = YouCookQADataset(image_path='.', split='training', slide_window_size=480)
    #img_feat, seg, _, q, _, ch, _, num = dataset[1]
    #print(mask)
    img_feat, seg, num, q, _, ch, _, num = dataset[21100]
    print(len(dataset.vocab))
    print(len(dataset))
    #print(q)
    #print(num)
    #print(ch)
    #print(dataset.answer_cls)
    exit(0)
    #print(mask)
    #print(ch)
    #print(ans_idx)
    #print(img_feat.shape)

