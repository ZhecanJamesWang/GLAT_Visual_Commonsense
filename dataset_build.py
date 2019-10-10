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
import utils
from tqdm import tqdm
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

# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_root, binary=True)

total_data = {}
total_data['input_embed'] = []
total_data['adj'] = []
total_data['gt_embed_ali'] = []
total_data['mask_idx'] = []

encoder_path = "./model/encoder_bpe_40000.json"
bpe_path = "./model/vocab_40000.bpe"
text_encoder = utils.TextEncoder(encoder_path, bpe_path)
encoder = text_encoder.encoder
new_categories = []
start_token = "<START>"
end_token = "<END>"
blank_token = "<blank>"
mask_token = "<MASK>"
new_categories += [blank_token]
new_categories += [start_token]
new_categories += [end_token]
new_categories += [mask_token]


subj_len = 0
obj_len = 0
max_length = 0

for idx in tqdm(range(len(rel_data[:1000]))):
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

        pred_len = len(text_encoder.encode([relationship['predicate']])[0])
        nodes += [relationship['predicate']]
        node_idxs.append('0')
        adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
        adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
        idx_pred = len(nodes) - 1 - nodes[::-1].index(relationship['predicate'])

        if relationship['subject']['object_id'] not in node_idxs:

            subj_len = len(text_encoder.encode([relationship['subject']['name']])[0])
            node_idxs.append(relationship['subject']['object_id'])
            nodes += [relationship['subject']['name']]
            adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
            adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
            idx_subject = node_idxs.index(relationship['subject']['object_id'])
        else:
            idx_subject = node_idxs.index(relationship['subject']['object_id'])

        if relationship['object']['object_id'] not in node_idxs:

            obj_len = len(text_encoder.encode([relationship['object']['name']])[0])
            node_idxs.append(relationship['object']['object_id'])
            nodes += [relationship['object']['name']]
            adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
            adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
            idx_object = node_idxs.index(relationship['object']['object_id'])
        else:
            idx_object = node_idxs.index(relationship['object']['object_id'])

        if pred_flg and sub_flg:
            adj[idx_subject][idx_pred] = 1
        if pred_flg and obj_flg:
            adj[idx_pred][idx_object] = 1
        max_length = max(pred_len, subj_len, obj_len, max_length)


    gt_embed = []
    input_embed = np.zeros((len(nodes), max_length), dtype=int)
    for category in new_categories:
        if category not in encoder.keys():
            text_encoder.decoder[len(encoder)] = category
            encoder[category] = len(encoder)

    for i, value in enumerate(nodes):
        len_node = len(text_encoder.encode([value])[0])
        # pdb.set_trace()
        input_embed[i][:len_node] = np.array(text_encoder.encode([value])[0])
        input_embed[i][len_node:] = np.array(encoder[blank_token])


    total_data['adj'].append(adj)
    total_data['input_embed'].append(input_embed)

# unifying the dimension of all node embeddings across graphs
for idx, input_embed in enumerate(total_data['input_embed']):
    if input_embed.shape[-1] < max_length:
        res_embed = np.ones((input_embed.shape[0], max_length-input_embed.shape[-1]), dtype=int) * encoder[blank_token]
        input_embed = np.concatenate((input_embed, res_embed), axis=-1)
    total_data['gt_embed_ali'].append(input_embed)

print('Data read done')
total_num = len(total_data['gt_embed_ali'])
train_num = int(total_num*0.8)
test_num = total_num - train_num

train_data = {}
train_data['gt_embed_ali'] = total_data['gt_embed_ali'][:train_num]
train_data['adj'] = total_data['adj'][:train_num]

test_data = {}
test_data['gt_embed_ali'] = total_data['gt_embed_ali'][train_num:]
test_data['adj'] = total_data['adj'][train_num:]

filename = 'train_VG_v1.pkl'
with open(filename,'wb') as f:
    pickle.dump(train_data, f)

filename = 'test_VG_v1.pkl'
with open(filename,'wb') as f:
    pickle.dump(test_data, f)
