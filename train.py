from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


# from pygcn.utils import load_data, accuracy
from data import VG_data
from torch.utils.data import DataLoader
from models import Ensemble_encoder
import pdb
import os
import utils
import torch.optim.lr_scheduler as lr_scheduler
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16,
#                     help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ToDo: hyperparameters connect with command line later
args.GAT_num = 1
args.Trans_num = 1
args.fea_dim = 300
args.nhid_gat = 100
args.nhid_trans = 300
args.n_heads = 8
args.batch_size = 3
args.weight_decay = 1e-4

global blank_idx

device = 'cuda'

home_path = os.getcwd()

print(home_path)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
train_dataset = VG_data(status='train', data_root=os.path.join(home_path, 'data'))

blank_idx = train_dataset.get_blank()


def my_collate(batch):
    max_length = 0
    for item in batch:
        max_length = max(max_length, item[0].size(0))
    # print('max length in batch is', max_length)
    max_length = max_length + 1
    gt_embeds = []
    input_embeds = []
    adjs = []
    input_masks = []
    pad_masks = []
    for i, (gt_embed, input_embed, adj, input_mask) in enumerate(batch):
        # print(i)
        gt_embeds.append(torch.cat((gt_embed, blank_idx*torch.ones(max_length-gt_embed.size(0), 1, dtype=torch.long)), 0).unsqueeze(0))
        input_embeds.append(torch.cat((input_embed, blank_idx*torch.ones(max_length-input_embed.size(0), 1, dtype=torch.long)), 0).unsqueeze(0))
        # pdb.set_trace()
        new_adj = torch.cat((adj, torch.zeros(max_length-adj.size(0), adj.size(1), dtype=torch.float)), 0)
        new_adj = torch.cat((new_adj, torch.zeros(new_adj.size(0), max_length-new_adj.size(1), dtype=torch.float)), 1)
        # pdb.set_trace()
        adjs.append(new_adj.unsqueeze(0))
        input_masks.append(torch.cat((input_mask, torch.zeros(max_length-input_mask.size(0),1, dtype=torch.long)), 0).unsqueeze(0))
        # pdb.set_trace()
        pad_masks.append(torch.cat((torch.zeros_like(gt_embed), torch.ones(max_length-gt_embed.size(0), 1, dtype=torch.long)), 0).unsqueeze(0))
    gt_embeds = torch.cat(gt_embeds, 0)
    input_embeds = torch.cat(input_embeds, 0)
    adjs = torch.cat(adjs, 0)
    input_masks = torch.cat(input_masks, 0)
    pad_masks = torch.cat(pad_masks, 0)
    return [gt_embeds, input_embeds, adjs, input_masks, pad_masks]


# Zhecan Wang 4:24 PM
# def my_collate(batch):
#     max_length = 0
#     for item in batch:
#         max_length = max(max_length, item[0].size(0))
#     # print(‘max length in batch is’, max_length)
#     gt_embeds = []
#     input_embeds = []
#     adjs = []
#     input_masks = []
#     for i, (gt_embed, input_embed, adj, input_mask) in enumerate(batch):
#         gt_embeds.append(torch.cat((gt_embed, blank_idx*torch.ones((max_length-gt_embed.size(0), 1), dtype=torch.long)), 0).unsqueeze(0))
#         input_embeds.append(torch.cat((input_embed, blank_idx*torch.ones((max_length-input_embed.size(0), 1), dtype=torch.long)), 0).unsqueeze(0))
#         new_adj = torch.cat((adj, torch.zeros((max_length-adj.size(0), adj.size(1)), dtype=torch.float)), 0)
#         new_adj = torch.cat((new_adj, torch.zeros((new_adj.size(0), max_length-new_adj.size(1)), dtype=torch.float)), 1)
#         adjs.append(new_adj.unsqueeze(0))
#         input_masks.append(torch.cat((input_mask, torch.zeros((max_length-input_mask.size(0),1), dtype=torch.long)), 0).unsqueeze(0))
#     gt_embeds = torch.cat(gt_embeds, 0)
#     input_embeds = torch.cat(input_embeds, 0)
#     adjs = torch.cat(adjs, 0)
#     input_masks = torch.cat(input_masks, 0)
#     return [gt_embeds, input_embeds, adjs, input_masks]

def get_gt_edge(pad_masks, adj):
    '''

    :param pad_masks: (B, N)
    :param adj:  (B, N, N)
    :return: pos_mask: (B*N*N)
             neg_mask: (B*N*N)
             bal_neg_mask: (B*N*N)
             effective_mask: (B*N*N) 1 denote no-padding
    '''
    B = pad_masks.size(0)
    N = pad_masks.size(1)
    # D = 1

    effective_mask = torch.mul((1-pad_masks).unsqueeze(-1), (1-pad_masks).unsqueeze(1))
    effective_mask = effective_mask.view(B, N*N).view(-1)
    pos_mask = adj.view(B, N*N).view(-1)
    neg_mask = (1-pos_mask).long()
    neg_mask = torch.mul(neg_mask, effective_mask)
    bal_neg_mask = copy.deepcopy(neg_mask)
    if len(torch.nonzero(neg_mask)) > len(torch.nonzero(pos_mask)):
        num_pos = len(torch.nonzero(pos_mask))
        bal_neg_mask[torch.nonzero(neg_mask)[num_pos:]] = 0

    return torch.nonzero(pos_mask).squeeze(-1), torch.nonzero(neg_mask).squeeze(-1), torch.nonzero(bal_neg_mask).squeeze(-1), torch.nonzero(effective_mask).squeeze(-1)

    # torch.cat((pad_masks.repeat(1, N).view(B, N * N), pad_masks.repeat(1, N, 1)), dim=2).view(B, N, -1, 2 * D)





train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=my_collate)
test_dataset = VG_data(status='test', data_root=os.path.join(home_path, 'data'))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=my_collate)


vocab_num = train_dataset.vocabnum()

# Model and optimizer
model = Ensemble_encoder(vocab_num=vocab_num,
                         feat_dim=args.fea_dim,
                         nhid_gat=args.nhid_gat,
                         nhid_trans=args.nhid_trans,
                         dropout=args.dropout,
                         nheads=args.n_heads,
                         alpha=args.alpha,
                         GAT_num=args.GAT_num,
                         Trans_num=args.Trans_num,
                         blank=blank_idx)

model = torch.nn.DataParallel(model)
model = model.to(device=device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 150], gamma=0.5)

cri_rec = torch.nn.CrossEntropyLoss()
cri_rec = cri_rec.to(device=device)

cri_con = torch.nn.CrossEntropyLoss()
cri_con = cri_con.to(device=device)


blank_idx = train_dataset.get_blank()


def train(epoch):
    model.train()
    t = time.time()
    loss_total_rec = 0
    loss_totoal_con = 0
    num_sample = 0
    node_acc = utils.Counter()
    edge_acc = utils.Counter()

    for i, (gt_embed, input_embed, adj, input_mask, pad_masks) in enumerate(train_loader):

        # pdb.set_trace()

        input_embed = input_embed.to(device=device)
        adj = adj.to(device=device)
        gt_embed = gt_embed.to(device=device)
        pad_masks = pad_masks.to(device=device)
        # input_mask = input_mask.to(device=device)

        # keep the last dimension for word length option
        input_embed = input_embed.squeeze(-1)
        gt_embed = gt_embed.squeeze(-1)
        pad_masks = pad_masks.squeeze(-1)

        if adj.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:
            pred_label, pred_connect = model(input_embed, adj)
            pred_label = pred_label.view(-1, pred_label.size(-1))
            # gt_embed = gt_embed.squeeze(-1).view(-1)
            gt_embed = gt_embed.view(-1)
            pos_mask, neg_mask, bal_neg_mask, effective_mask = get_gt_edge(pad_masks, adj)
            pred_connect_train = torch.cat((pred_connect[pos_mask], pred_connect[bal_neg_mask]), 0)
            gt_edge_train = torch.cat((torch.ones(len(pos_mask)), torch.zeros(len(bal_neg_mask))), 0).long()

            pred_connect_eff = torch.cat((pred_connect[pos_mask], pred_connect[neg_mask]), 0)
            gt_edge_eff = torch.cat((torch.ones(len(pos_mask)), torch.zeros(len(neg_mask))), 0).long()

            # torch.cat((pred_connect[pos_mask], pred_connect[neg_mask]), 0)
            # torch.cat((torch.ones(num_list[0]), torch.zeros(num_list[1])), 0).long()
            # pdb.set_trace()
            pred_label_eff = pred_label[pad_masks.view(-1)==0]
            gt_embed_eff = gt_embed[pad_masks.view(-1)==0]
            loss_rec = cri_rec(pred_label_eff, gt_embed_eff)
            loss_con = cri_con(pred_connect_train, gt_edge_train.to(device=device))
            loss = loss_rec + loss_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_sample += 1
            loss_total_rec += loss_rec.item()
            loss_totoal_con += loss_con.item()

            node_acc.add(pred_label_eff, gt_embed_eff)
            edge_acc.add(pred_connect_eff, gt_edge_eff)

            # if not args.fastmode:
            #     # Evaluate validation set performance separately,
            #     # deactivates dropout during validation run.
            #     model.eval()
            #     output = model(features, adj)

            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = accuracy(output[idx_val], labels[idx_val])
            if (i+1) % 1000 == 0:
                print('Train Epoch: {:04d} [{}/{}]'.format(epoch, i, len(train_loader)),
                      'loss_rec: {:.4f}'.format(loss_total_rec/num_sample),
                      'loss_con: {:.4f}'.format(loss_totoal_con/num_sample),
                      'node_acc_train: {:.4f}'.format(node_acc.mean()),
                      'edge_acc_train: {:.4f}'.format(edge_acc.mean()),
                      # 'loss_val: {:.4f}'.format(loss_val.item()),
                      # 'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    print('Train Epoch Finished: {:04d}'.format(epoch),
      'loss_rec: {:.4f}'.format(loss_total_rec/num_sample),
      'loss_con: {:.4f}'.format(loss_totoal_con/num_sample),
          'node_acc_train: {:.4f}'.format(node_acc.mean()),
          'edge_acc_train: {:.4f}'.format(edge_acc.mean()),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s'.format(time.time() - t))


def test(epoch):
    model.eval()
    t = time.time()

    loss_total_rec = 0
    loss_totoal_con = 0
    num_sample = 0
    node_acc = utils.Counter()
    edge_acc = utils.Counter()

    for i, (gt_embed, input_embed, adj, input_mask, pad_masks) in enumerate(test_loader):
        input_embed = input_embed.to(device=device)
        adj = adj.to(device=device)
        gt_embed = gt_embed.to(device=device)
        pad_masks = pad_masks.to(device=device)

        # input_mask = input_mask.to(device=device)

        input_embed = input_embed.squeeze(-1)
        gt_embed = gt_embed.squeeze(-1)
        pad_masks = pad_masks.squeeze(-1)

        if adj.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:
            pred_label, pred_connect = model(input_embed, adj)
            pred_label = pred_label.view(-1, pred_label.size(-1))
            # gt_embed = gt_embed.squeeze(-1).view(-1)
            gt_embed = gt_embed.view(-1)
            pos_mask, neg_mask, bal_neg_mask, effective_mask = get_gt_edge(pad_masks, adj)
            # pdb.set_trace()

            pred_connect_eff = torch.cat((pred_connect[pos_mask], pred_connect[neg_mask]), 0)
            gt_edge_eff = torch.cat((torch.ones(len(pos_mask)), torch.zeros(len(neg_mask))), 0).long()

            pred_label_eff = pred_label[pad_masks.view(-1)==0]
            gt_embed_eff = gt_embed[pad_masks.view(-1)==0]
            loss_rec = cri_rec(pred_label_eff, gt_embed_eff)
            loss_con = cri_con(pred_connect_eff, gt_edge_eff.to(device=device))
            # loss = loss_rec + loss_con

            num_sample += 1
            loss_total_rec += loss_rec.item()
            loss_totoal_con += loss_con.item()

            node_acc.add(pred_label_eff, gt_embed_eff)
            edge_acc.add(pred_connect_eff, gt_edge_eff)

    print('Test Epoch Finished: {:04d}'.format(epoch),
      'loss_rec: {:.4f}'.format(loss_total_rec/num_sample),
      'loss_con: {:.4f}'.format(loss_totoal_con/num_sample),
          'node_acc_test: {:.4f}'.format(node_acc.mean()),
          'edge_acc_test: {:.4f}'.format(edge_acc.mean()),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s'.format(time.time() - t))

    # print("Test set results:",
    #         'loss_rec: {:.4f}'.format(loss_rec.item()),
    #         'loss_con: {:.4f}'.format(loss_con.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    scheduler.step()

    train(epoch)

    # if (epoch+1) % 5 == 0:
    test(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()


# Todo:
# 1. loss
# 2. dimension matching
# 3. saving & loading vocab info
# 4. (argument of Attention kvq)