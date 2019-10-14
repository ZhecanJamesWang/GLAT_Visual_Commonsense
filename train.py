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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
args.batch_size = 5

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
    gt_embeds = []
    input_embeds = []
    adjs = []
    input_masks = []
    for i, (gt_embed, input_embed, adj, input_mask) in enumerate(batch):
        gt_embeds.append(torch.cat((gt_embed, blank_idx*torch.ones((max_length-gt_embed.size(0), 1), dtype=torch.long)), 0).unsqueeze(0))
        input_embeds.append(torch.cat((input_embed, blank_idx*torch.ones((max_length-input_embed.size(0), 1), dtype=torch.long)), 0).unsqueeze(0))
        new_adj = torch.cat((adj, torch.zeros((max_length-adj.size(0), adj.size(1)), dtype=torch.float)), 0)
        new_adj = torch.cat((new_adj, torch.zeros((new_adj.size(0), max_length-new_adj.size(1)), dtype=torch.float)), 1)
        adjs.append(new_adj.unsqueeze(0))
        input_masks.append(torch.cat((input_mask, torch.zeros((max_length-input_mask.size(0),1), dtype=torch.long)), 0).unsqueeze(0))
    gt_embeds = torch.cat(gt_embeds, 0)
    input_embeds = torch.cat(input_embeds, 0)
    adjs = torch.cat(adjs, 0)
    input_masks = torch.cat(input_masks, 0)
    return [gt_embeds, input_embeds, adjs, input_masks]


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, drop_last=False, collate_fn=my_collate)
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
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

cri_rec = torch.nn.CrossEntropyLoss()
# cri_rec = torch.nn.MSELoss()
cri_rec = cri_rec.to(device = device)

cri_con = torch.nn.CrossEntropyLoss()
cri_con = cri_con.to(device = device)

def train(epoch):
    model.train()
    t = time.time()
    loss_total_rec = 0
    loss_totoal_con = 0
    num_sample = 0

    for i, (gt_embed, input_embed, adj, input_mask) in enumerate(train_loader):

        input_embed = input_embed.to(device=device)
        adj = adj.to(device=device)
        gt_embed = gt_embed.to(device=device)
        input_mask = input_mask.to(device=device)

        input_embed = input_embed.squeeze(-1)
        gt_embed = gt_embed.squeeze(-1)

        if adj.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:
            pred_label, pred_connect, num_list = model(input_embed, adj)

            pred_label = pred_label.view(-1, pred_label.size(-1))
            # gt_embed = gt_embed.squeeze(-1).view(-1)
            gt_embed = gt_embed.view(-1)

            loss_rec = cri_rec(pred_label, gt_embed)
            loss_con = cri_con(pred_connect,torch.cat((torch.ones(num_list[0]),torch.zeros(num_list[1])), 0).long().to(device=device))
            loss = loss_rec + loss_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_sample += 1
            loss_total_rec += loss_rec.item()
            loss_totoal_con += loss_con.item()

            # if not args.fastmode:
            #     # Evaluate validation set performance separately,
            #     # deactivates dropout during validation run.
            #     model.eval()
            #     output = model(features, adj)

            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = accuracy(output[idx_val], labels[idx_val])
            if (i+1) % 100 == 0:
                print('Epoch: {:04d} [{}/83858]'.format(epoch, i),
                      'loss_rec: {:.4f}'.format(loss_total_rec/num_sample),
                      'loss_con: {:.4f}'.format(loss_totoal_con/num_sample),
                      # 'acc_train: {:.4f}'.format(acc_train.item()),
                      # 'loss_val: {:.4f}'.format(loss_val.item()),
                      # 'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    print('Epoch Finished: {:04d}'.format(epoch),
      'loss_rec: {:.4f}'.format(loss_total_rec/num_sample),
      'loss_con: {:.4f}'.format(loss_totoal_con/num_sample),
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

    for i, (gt_embed, input_embed, adj, input_mask) in enumerate(test_loader):
        input_embed = input_embed.to(device=device)
        adj = adj.to(device=device)
        gt_embed = gt_embed.to(device=device)
        input_mask = input_mask.to(device=device)

        input_embed = input_embed.squeeze(-1)
        gt_embed = gt_embed.squeeze(-1)
        # print(input_embed.size())
        # print(adj.size())
        # print(gt_embed.size())

        # input_embed = input_embed.to(device=device)
        # adj = adj.to(device=device)
        # gt_embed = gt_embed.to(device=device)

        if adj.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:
            pred_label, pred_connect, num_list = model(input_embed, adj)
            pred_label = pred_label.view(-1, pred_label.size(-1))
            gt_embed = gt_embed.view(-1)

            loss_rec = cri_rec(pred_label, gt_embed)
            loss_con = cri_con(pred_connect,torch.cat((torch.ones(num_list[0]),torch.zeros(num_list[1])), 0).long().to(device=device))

            num_sample += 1
            loss_total_rec += loss_rec.item()
            loss_totoal_con += loss_con.item()

    print('Epoch Finished: {:04d}'.format(epoch),
      'loss_rec: {:.4f}'.format(loss_total_rec/num_sample),
      'loss_con: {:.4f}'.format(loss_totoal_con/num_sample),
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
    train(epoch)
    if (epoch+1) % 5 == 0:
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