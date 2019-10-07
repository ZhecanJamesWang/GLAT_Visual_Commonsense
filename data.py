import numpy as np
import scipy.sparse as sp
import torch
import json
import os
import pdb
import time
from torch.utils.data import Dataset
import gensim
import pickle
# from nltk.corpus import stopwords

class VG_data(Dataset):
    def __init__(self, data_root='/home/haoxuan/code/pygcn/', status='train'):
        super(VG_data, self).__init__()

        # self.rel_root = os.path.join(data_root, 'relationships.json')
        # self.atr_root = os.path.join(data_root, 'attributes.json')
        self.status = status
        # self.w2v_root = os.path.join(w2v_root, 'GoogleNews-vectors-negative300.bin')

        # with open(self.rel_root) as f:
        #     self.rel_data = json.load(f)

        if self.status == 'train':
            self.data_root = os.path.join(data_root, 'train_VG.pkl')
        else:
            self.data_root = os.path.join(data_root, 'test_VG.pkl')

        self.data = pickle.load(open(self.data_root,'rb'), encoding='latin')

        print('{} data num: {}'.format(status, len(self.data['input_embed'])))

    def __getitem__(self, idx):
        input_embed = self.data['input_embed'][idx]
        adj = self.data['adj'][idx]
        gt_embed = self.data['gt_embed'][idx]
        mask_idx = self.data['mask_idx'][idx]
        # pdb.set_trace()
        # print(idx)

        return torch.from_numpy(np.array(input_embed)).float(), torch.from_numpy(np.array(adj)).float(),torch.from_numpy(np.array(gt_embed)).float(), mask_idx

    def __len__(self):
        return len(self.data['input_embed'])


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

if __name__ == '__main__':
    time0 = time.time()
    train_dataset = VG_data(status='train')
    time1 = time.time()
    print('Load model and data{}'.format(time1-time0))
    input_embed, adj, gt_embed, mask_dix = train_dataset.__getitem__(0)
    print('build data graph and embedding{}'.format(time.time()-time1))
    pdb.set_trace()


