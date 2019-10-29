import numpy as np
import scipy.sparse as sp
import torch
import json
import os
import pdb
import time
from torch.utils.data import Dataset
import pickle
import utils
import math
import copy
import random
# from nltk.corpus import stopwords
import random
random.seed(3)


class VG_data(Dataset):
    def __init__(self, data_root='/data/', status='train'):
        super(VG_data, self).__init__()

        # data_num = 10000
        # name = '_'.join([str(10000), str(2000)])
        name = '_'.join(['full', 'full'])

        with open(os.path.join(data_root, 'vocab_clean_{}_image_id.pkl'.format(name)), 'rb') as f:
            self.vocab = pickle.load(f, encoding='latin')
        # self.vocab_encoder = self.vocab['encoder']
        # self.vocab_decoder = self.vocab['decoder']
        # self.vocab_num = len(self.vocab_encoder.keys())
        self.vocab_num = len(self.vocab)
        self.mask_prob = 0.1

        print('vocabulary number', self.vocab_num)

#         encoder_path = "./model/encoder_bpe_40000.json"
#         bpe_path = "./model/vocab_40000.bpe"
#         self.text_encoder = utils.TextEncoder(encoder_path, bpe_path)
#         new_categories = []
#         start_token = "<START>"
#         end_token = "<END>"
#         blank_token = "<blank>"
#         mask_token = "<MASK>"
#         new_categories += [blank_token]
#         new_categories += [start_token]
#         new_categories += [end_token]
#         new_categories += [mask_token]
#         for category in new_categories:
#             if category not in self.text_encoder.encoder.keys():
#                 self.text_encoder.decoder[len(self.text_encoder.encoder)] = category
#                 self.text_encoder.encoder[category] = len(self.text_encoder.encoder)

        # self.rel_root = os.path.join(data_root, 'relationships.json')
        # self.atr_root = os.path.join(data_root, 'attributes.json')
        self.status = status
        # self.w2v_root = os.path.join(w2v_root, 'GoogleNews-vectors-negative300.bin')

        # with open(self.rel_root) as f:
        #     self.rel_data = json.load(f)

        if self.status == 'train':
            self.data_root = os.path.join(data_root, 'train_VG_clean_{}_image_id.pkl'.format(name))
        else:
            self.data_root = os.path.join(data_root, 'test_VG_clean_{}_image_id.pkl'.format(name))

        with open(self.data_root,'rb') as f:
            self.data = pickle.load(f, encoding='latin')

        print('{} data num: {}'.format(status, len(self.data['gt_embed_ali'])))

    def __getitem__(self, idx):
        # print("idx: ", idx)
        gt_embed = self.data['gt_embed_ali'][idx]
        adj = self.data['adj'][idx]

        # print("gt_embed: ", gt_embed)
        # print("gt_embed.shape: ", gt_embed.shape)

        names = []
        for [idx_node] in gt_embed:
            names.append(self.vocab[idx_node])

        mask_num = math.ceil(gt_embed.shape[0] * self.mask_prob)
        mask_idx = random.sample(range(0, gt_embed.shape[0]), mask_num)
        input_mask = np.zeros((gt_embed.shape[0], 1), dtype=int)
        input_mask[mask_idx] = 1

        input_embed = copy.deepcopy(gt_embed)

        if len(mask_idx) != 0:
            input_embed[mask_idx] = np.array(self.vocab.index("<MASK>"))
            # input_embed[mask_idx] = np.array(self.vocab.index("<MASK>"+"</w>"))

        img_id = self.data['img_id'][idx]

        # input_embed[mask_idx] = np.array(self.vocab_encoder["<MASK>"+"</w>"])
        # input_embed[mask_idx] = np.array(self.text_encoder.encoder["<MASK>"])

        # pdb.set_trace()

        # gt_embed = self.data['gt_embed'][idx]
        # mask_idx = self.data['mask_idx'][idx]
        # pdb.set_trace()
        # print(idx)

        return torch.from_numpy(np.array(gt_embed)).long(), torch.from_numpy(np.array(input_embed)).long(),\
               torch.from_numpy(np.array(adj)).float(), torch.from_numpy(np.array(input_mask)), names, img_id

    def __len__(self):
        return len(self.data['gt_embed_ali'])

    def idx_to_name(self, idxs):
        return self.vocab[idxs]

    def vocabnum(self):
        return self.vocab_num

    def get_blank(self):
        # return self.vocab_encoder["<blank>" + "</w>"]
        return self.vocab.index("<blank>")

    def get_stats(self):
        len_data = [0]
        for i, data in enumerate(self.data['gt_embed_ali']):
            len_data += [data.shape[0]]
        len_data = np.asarray(len_data)
        print('mean length:', np.mean(len_data))
        print('median length:', np.median(len_data))
        print('max length:', np.max(len_data))
        len_data_over = len_data[len_data>50]
        pdb.set_trace()
        return len_data

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

if __name__ == '__main__':
    home_path = os.getcwd()

    time0 = time.time()
    train_dataset = VG_data(status='train', data_root=os.path.join(home_path,'data'))
    time1 = time.time()
    train_dataset.get_stats()
    # print('Load model and data{}'.format(time1-time0))
    gt_embed, input_embed, adj, input_mask = train_dataset.__getitem__(0)
    # print('build data graph and embedding{}'.format(time.time()-time1))
    pdb.set_trace()


