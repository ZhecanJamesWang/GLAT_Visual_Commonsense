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


class VG_data(Dataset):
    def __init__(self, data_root='/data/', status='train'):
        super(VG_data, self).__init__()

        # data_num = 10000
        # name = '_'.join([str(10000), str(2000)])
        # name = '_'.join(['full', 'full'])
        name = '_'.join(['20000', '1000', '5000'])

        with open(os.path.join(data_root, 'ind_to_classes_{}.pkl'.format(name)), 'rb') as f:
            self.ind_to_entities = pickle.load(f, encoding='latin')
            self.ind_to_entities.append("<MASK>")
            self.ind_to_entities.append("<blank>")

        with open(os.path.join(data_root, 'ind_to_predicates_{}.pkl'.format(name)), 'rb') as f:
            self.ind_to_predicates = pickle.load(f, encoding='latin')
            self.ind_to_predicates.append("<MASK>")
            # self.ind_to_predicates.append("<blank>")



        # self.vocab_encoder = self.vocab['encoder']
        # self.vocab_decoder = self.vocab['decoder']
        # self.vocab_num = len(self.vocab_encoder.keys())
        # self.vocab_num = len(self.vocab)
        self.mask_prob = 0.5
        self.noise_prob = 1

        # print('vocabulary number', self.vocab_num)

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
            self.data_root = os.path.join(data_root, 'train_VG_kern_{}.pkl'.format(name))
        elif self.status == 'eval':
            self.data_root = os.path.join(data_root, 'eval_VG_kern_{}.pkl'.format(name))
        elif self.status == 'test':
            self.data_root = os.path.join(data_root, 'test_VG_kern_{}.pkl'.format(name))

        with open(self.data_root,'rb') as f:
            self.data = pickle.load(f, encoding='latin')

        print('{} data num: {}'.format(status, len(self.data['node_name'])))

    def __getitem__(self, idx):
        node_class = self.data['node_class'][idx]
        adj = self.data['adj'][idx]
        node_type = self.data['node_type'][idx]

        entity_num = sum(node_type)
        predicate_num = len(node_type) - entity_num

        # mask for predicate
        mask_num_predicate = math.ceil(predicate_num * self.mask_prob)
        mask_idx_predicate = random.sample(list(np.where(node_type == 0)[0]), mask_num_predicate)
        input_mask = np.zeros(node_class.shape[0], dtype=int)  # 0-regular   1-mask/noise
        input_mask[mask_idx_predicate] = 1

        input_class = copy.deepcopy(node_class)

        if len(mask_idx_predicate) != 0:

            noise_num = math.ceil(len(mask_idx_predicate) * self.noise_prob)
            mask_idx_copy = copy.deepcopy(mask_idx_predicate)
            noise_idx = []

            for i in range(noise_num):
                noise_idx += [mask_idx_copy.pop()]

            for idx_node in mask_idx_predicate:
                if idx_node not in noise_idx:
                    input_class[idx_node] = self.ind_to_predicates.index("<MASK>")

                else:
                    target_type = node_type[idx_node]
                    if target_type == 0:
                        # pdb.set_trace()
                        input_class[idx_node] = random.sample(list(range(len(self.ind_to_predicates))), 1)[0]
                    else:
                        input_class[idx_node] = random.sample(list(range(len(self.ind_to_entities))), 1)[0]

        return torch.from_numpy(np.array(node_class)).long(), torch.from_numpy(np.array(input_class)).long(),\
               torch.from_numpy(np.array(adj)).float(), torch.from_numpy(np.array(input_mask)).long(), torch.from_numpy(np.array(node_type)).long()

    def __len__(self):
        return len(self.data['node_class'])

    # def vocab_num(self):
    #     return self.vocab_num
    def vocab_num(self):
        return len(self.ind_to_predicates), len(self.ind_to_entities)

    def get_blank(self):
        # return self.vocab_encoder["<blank>" + "</w>"]
        return self.ind_to_entities.index("<blank>")

    def get_stats(self):
        len_data = [0]
        for i, data in enumerate(self.data['node_class']):
            len_data += [data.shape[0]]
        len_data = np.asarray(len_data)
        print('mean length:', np.mean(len_data))
        print('median length:', np.median(len_data))
        print('max length:', np.max(len_data))
        len_data_over = len_data[len_data>50]
        # pdb.set_trace()
        return len_data


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

if __name__ == '__main__':
    # home_path = os.getcwd()
    home_path = '/home/haoxuan/code/KERN'

    time0 = time.time()
    train_dataset = VG_data(status='train', data_root=os.path.join(home_path,'data'))
    time1 = time.time()
    train_dataset.get_stats()
    # print('Load model and data{}'.format(time1-time0))
    node_class, input_class, adj, input_mask, node_type = train_dataset.__getitem__(0)
    # print('build data graph and embedding{}'.format(time.time()-time1))
    # pdb.set_trace()


