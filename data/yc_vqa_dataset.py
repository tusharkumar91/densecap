import torch
import torch.utils.data
import os
import json
import tqdm
import pickle
import numpy as np
from nltk import word_tokenize
import math

class YouCookQA(torch.utils.data.Dataset):
    def __init__(self, image_path, split='training', slide_window_size=480):
        self.split = split
        self.slide_window_size = slide_window_size
        self.vocab = {}
        self.qa_ans_vid_data = []
        self.split_path = os.path.join(image_path, split)
        self.initialize_vocab()
        self.initialize_data()

    def initialize_data(self):
        with open("data/final_db_3K_combined_without_yesno.json") as f:
            qa_data = json.load(f)
        for qa in qa_data["QAPairs"]:
            if qa["split"] == self.split:
                vid = qa["video_id"]
                q = np.zeros((36))
                q_tokens = word_tokenize(qa["question"])
                idx = -1
                for token in q_tokens:
                    q[idx] = self.vocab[token]
                    idx = idx - 1
                answers = np.zeros((5, 36))
                answer_tokens = word_tokenize(qa["answer"])

                for token in answer_tokens:
                    answers[0][idx] = self.vocab[token]
                    idx = idx - 1
                for opt_idx, opt in enumerate(qa["alternatives"]):
                    idx = -1
                    opt_tokens = word_tokenize(opt)
                    for token in opt_tokens:
                        answers[opt_idx + 1][idx] = self.vocab[token]
                        idx = idx - 1
                self.qa_ans_vid_data.append({
                    "question" : q,
                    "choices" : answers,
                    "ans_idx" : 0,
                    "vid" : vid
                })
        print('Total {} qa pairs in {}'.format(len(self.qa_ans_vid_data), self.split))

    def initialize_vocab(self):
        dict_dir = 'data/vocab_3K_combined_without_yesno.json'
        with open(dict_dir) as f:
            word_to_idx = json.load(f)
        self.vocab = word_to_idx
        # for word in word_to_idx:
        #     idx_to_word[word_to_idx[word]] = word
        # self.old_idx_to_word = idx_to_word
        # print(len(self.old_idx_to_word))

    def __getitem__(self, idx):
        qa_ans_vid_data_idx = self.qa_ans_vid_data[idx]
        q = qa_ans_vid_data_idx["question"]
        ch = qa_ans_vid_data_idx["choices"]
        ans_idx = qa_ans_vid_data_idx["ans_idx"]
        vid = qa_ans_vid_data_idx["vid"]

        video_prefix = os.path.join(self.split_path, vid)
        resnet_feat = torch.from_numpy(
            np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
        img_feat = torch.FloatTensor(np.zeros((self.slide_window_size,
                                               resnet_feat.size(1) + bn_feat.size(1))))
        total_frame = bn_feat.size(0)
        torch.cat((resnet_feat, bn_feat), dim=1,
                  out=img_feat[:min(total_frame, self.slide_window_size)])

        return img_feat, q, ch, ans_idx

    def __len__(self):
        # if self.split == "validation":
        #    return len(q_ids)
        #    #return len(self.mat_id)
        # elif self.split == "training":
        #    return len(train_q_ids)
        # else:
        return len(self.qa_ans_vid_data)


if __name__ == "__main__":
    dataset = YouCookQA(image_path=None, split='training', slide_window_size=480)
    img_feat, q, ch, ans_idx = dataset[0]
    print(q)
    print(ch)
    print(ans_idx)
    print(img_feat.shape)

