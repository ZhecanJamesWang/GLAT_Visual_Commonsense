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
data_root='/home/haoxuan/data/VG/'
w2v_root='/home/haoxuan/data/'
status='train'

rel_root = os.path.join(data_root, 'relationships.json')
atr_root = os.path.join(data_root, 'attributes.json')
status = status
w2v_root = os.path.join(w2v_root, 'GoogleNews-vectors-negative300.bin')

with open(rel_root) as f:
    rel_data = json.load(f)

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_root, binary=True)  

total_data = {}
total_data['input_embed'] = []
total_data['adj'] = []
total_data['gt_embed'] = []
total_data['mask_idx'] = []


for idx in range(len(rel_data)):
    single_rel_data = rel_data[idx]['relationships']
    if len(single_rel_data) == 0:
        continue
    nodes = []
    node_idxs = []
    adj = np.zeros(shape=(0, 0))

    for count, relationship in enumerate(single_rel_data):
        pred_flg = 1
        sub_flg = 1
        obj_flg = 1
        try:
            word2vec_model[relationship['predicate']]
        except:
            pred_flg = 0
        else:
            nodes.append(relationship['predicate'])
            node_idxs.append('0')
            adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
            adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
            idx_pred = len(nodes) - 1 - nodes[::-1].index(relationship['predicate'])

        if relationship['subject']['object_id'] not in node_idxs:
            try:
                word2vec_model[relationship['subject']['name']]
            except:
                sub_flg = 0
            else:
                node_idxs.append(relationship['subject']['object_id'])
                nodes.append([relationship['subject']['name']])
                adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
                adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
                idx_subject = node_idxs.index(relationship['subject']['object_id'])
        else:
            idx_subject = node_idxs.index(relationship['subject']['object_id'])

        if relationship['object']['object_id'] not in node_idxs:
            try:
                word2vec_model[relationship['object']['name']]
            except:
                obj_flg = 0
            else:
                node_idxs.append(relationship['object']['object_id'])
                nodes.append([relationship['object']['name']])
                adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
                adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
                idx_object = node_idxs.index(relationship['object']['object_id'])
        else:
            idx_object = node_idxs.index(relationship['object']['object_id'])

        if pred_flg and sub_flg:
            adj[idx_subject][idx_pred] = 1
        if pred_flg and obj_flg:
            adj[idx_pred][idx_object] = 1

    gt_embed = []
    input_embed = []
    try:
        mask_idx = np.random.randint(0, len(nodes))
    except:
        continue

    for i, value in enumerate(nodes):
        value = value[0] if type(value) == list else value
        gt_embed.append(word2vec_model[value])
        if i == mask_idx:
            input_embed.append(0.5 * np.ones(gt_embed[-1].shape))
        else:
            input_embed.append(word2vec_model[value])

    total_data['gt_embed'].append(gt_embed)
    total_data['adj'].append(adj)
    total_data['input_embed'].append(input_embed)
    total_data['mask_idx'].append(mask_idx)

    # try:
    #     input_emb.append(word2vec_model[value])
    # except:
    #     value_list = value.split(" ")
    #     value_emb = []
    #     for i in value_list:
    #         try:
    #             input_emb.append(word2vec_model[value])
    #         except:
    #             continue
    #     if len(value_emb) == 0:
    #         print(' {} has no corresponding embedding'.format(value))
    #         # value_emb.append(self.word2vec_model[i])
    #     input_emb.append(np.mean(np.array(value_emb), axis=0))

         # torch.from_numpy(input_emb).float(), torch.from_numpy(np.array(adj))

print('Data read done')
total_num = len(total_data['input_embed'])
train_num = int(total_num*0.8)
test_num = total_num - train_num

train_data = {}
train_data['input_embed'] = total_data['input_embed'][:train_num]
train_data['adj'] = total_data['adj'][:train_num]
train_data['gt_embed'] = total_data['gt_embed'][:train_num]
train_data['mask_idx'] = total_data['mask_idx'][:train_num]

test_data = {}
test_data['input_embed'] = total_data['input_embed'][train_num:]
test_data['adj'] = total_data['adj'][train_num:]
test_data['gt_embed'] = total_data['gt_embed'][train_num:]
test_data['mask_idx'] = total_data['mask_idx'][train_num:]


for i, adj in enumerate(train_data['adj']):
    if adj.shape[0] != adj.shape[1]:
        print (i)


pdb.set_trace()


filename = 'train_VG.pkl'
with open(filename,'wb') as f:
    pickle.dump(train_data, f)

filename = 'test_VG.pkl'
with open(filename,'wb') as f:
    pickle.dump(test_data, f)


# def encode_onehot(labels):
#     classes = set(labels)
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
#                     enumerate(classes)}
#     labels_onehot = np.array(list(map(classes_dict.get, labels)),
#                              dtype=np.int32)
#     return labels_onehot

# if __name__ == '__main__':
#     time0 = time.time()
#     train_dataset = VG_data(status='train')
#     time1 = time.time()
#     print('Load model and data{}'.format(time1-time0))
#     nodes, adj = train_dataset.__getitem__(0)
#     print('build data graph and embedding{}'.format(time.time()-time1))
#     pdb.set_trace()


