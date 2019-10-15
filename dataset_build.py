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
from tqdm import tqdm
import os
# from nltk.corpus import stopwords

home_path = os.environ['HOME']

# data_root = os.path.join(home_path, 'data/VG')
data_root = os.path.join(home_path, 'data/top_150_50_new')
filename = 'train.json'

status = 'train'

rel_root = os.path.join(data_root, filename)
# atr_root = os.path.join(data_root, 'attributes.json')
status = status

print("loading vg data: ", filename)

with open(rel_root) as f:
    rel_data = json.load(f)

print("finish loading vg data: ", filename)

total_data = {}
total_data['gt_embed'] = []
total_data['adj'] = []
total_data['gt_embed_ali'] = []
total_data['mask_idx'] = []
total_data['node_name'] = []
# total_data['input_embed'] = []
# total_data['adj'] = []
# total_data['gt_embed_ali'] = []
# total_data['mask_idx'] = []

encoder_path = "./model/encoder_bpe_40000.json"
bpe_path = "./model/vocab_40000.bpe"

print("loading: ", encoder_path, bpe_path)
text_encoder = utils.TextEncoder(encoder_path, bpe_path)
print("finish loading: ", encoder_path, bpe_path)

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
fix_vocabulary = 1

for idx in tqdm(range(len(rel_data[:5000]))):
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
        # if len(text_encoder.encode([relationship['predicate']])[0]) != 1 and relationship['predicate'] not in new_categories:
        if relationship['predicate'] not in new_categories:
            new_categories += [relationship['predicate'].lower()]

        pred_len = len(text_encoder.encode([relationship['predicate']])[0])
        nodes += [relationship['predicate'].lower()]
        node_idxs.append('0')
        adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
        adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
        idx_pred = len(nodes) - 1 - nodes[::-1].index(relationship['predicate'].lower())

        # original version ==> clean version
        # relationship['subject']['object_id'] ==> relationship['sub_id']
        # relationship['subject']['name'] ==> rel_data[idx]['objects'][relationship['sub_id']]['class']
        if relationship['sub_id'] not in node_idxs:
            # if len(text_encoder.encode([relationship['subject']['name']])[0]) != 1 and relationship['subject']['name'] not in new_categories:
            if rel_data[idx]['objects'][relationship['sub_id']]['class'] not in new_categories:
                new_categories += [rel_data[idx]['objects'][relationship['sub_id']]['class'].lower()]

            subj_len = len(text_encoder.encode([rel_data[idx]['objects'][relationship['sub_id']]['class']])[0])
            node_idxs.append(relationship['sub_id'])
            nodes += [rel_data[idx]['objects'][relationship['sub_id']]['class'].lower()]
            adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
            adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
            idx_subject = node_idxs.index(relationship['sub_id'])
        else:
            idx_subject = node_idxs.index(relationship['sub_id'])

        # original version ==> clean version
        # relationship['object']['object_id'] ==> relationship['obj_id']
        # relationship['object']['name'] ==> rel_data[idx]['objects'][relationship['obj_id']]['class']
        if relationship['obj_id'] not in node_idxs:
            # if len(text_encoder.encode([relationship['object']['name']])[0]) != 1 and relationship['object']['name'] not in new_categories:
            if rel_data[idx]['objects'][relationship['obj_id']]['class'] not in new_categories:
                new_categories += [rel_data[idx]['objects'][relationship['obj_id']]['class'].lower()]

            obj_len = len(text_encoder.encode([rel_data[idx]['objects'][relationship['obj_id']]['class']])[0])
            node_idxs.append(relationship['obj_id'])
            nodes += [rel_data[idx]['objects'][relationship['obj_id']]['class'].lower()]
            adj = np.append(adj, [adj.shape[1]*[0]], axis=0)
            adj = np.append(adj, np.array([adj.shape[0]*[0]]).transpose(), axis=1)
            idx_object = node_idxs.index(relationship['obj_id'])
        else:
            idx_object = node_idxs.index(relationship['obj_id'])

        if pred_flg and sub_flg:
            adj[idx_subject][idx_pred] = 1
        if pred_flg and obj_flg:
            adj[idx_pred][idx_object] = 1
        max_length = max(pred_len, subj_len, obj_len, max_length)

    total_data['adj'].append(adj)
    total_data['node_name'].append(nodes)

print('number of new categories', len(new_categories))

pdb.set_trace()

for category in new_categories:
    if category not in encoder.keys() and category+'</w>' not in encoder.keys():
        text_encoder.decoder[len(encoder)] = category + '</w>'
        encoder[category + '</w>'] = len(encoder)

if fix_vocabulary:
    max_length = 1

for graph_idx, nodes in enumerate(total_data['node_name']):
    input_embed = np.zeros((len(nodes), max_length), dtype=int)
    for node_idx, name in enumerate(nodes):
        if fix_vocabulary:
            input_embed[node_idx] = np.array(encoder[name]) if name in encoder.keys() else np.array(encoder[name+'</w>'])
        else:
            len_node = len(text_encoder.encode([name])[0])
            input_embed[:len_node] = np.array(text_encoder.encode([name])[0])
            input_embed[len_node:] = np.array(encoder[blank_token])
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

save_root = 'data/'
if not os.path.exists(save_root):
    os.mkdir(save_root)

vocab = {}
vocab['encoder'] = text_encoder.encoder
vocab['decoder'] = text_encoder.decoder

filename = os.path.join(save_root, 'train_VG_v3.pkl')
with open(filename, 'wb') as f:
    pickle.dump(train_data, f)

filename = os.path.join(save_root, 'test_VG_v3.pkl')
with open(filename, 'wb') as f:
    pickle.dump(test_data, f)

filename = os.path.join(save_root, 'vocab_v3.pkl')
with open(filename, 'wb') as f:
    pickle.dump(vocab, f)

