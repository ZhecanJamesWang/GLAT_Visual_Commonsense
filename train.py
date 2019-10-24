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
from models import Baseline, GLATNET
import pdb
import os
import utils
import torch.optim.lr_scheduler as lr_scheduler
import copy
import random
import datetime
from tensorboardX import SummaryWriter

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M")

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
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ToDo: hyperparameters connect with command line later
# ===================
# GLAT Model Additional Parameters
# args.struct = [1, 1, 1]
args.struct = [2, 2, 2]

args.nhid_glat_g = 300
args.nhid_glat_l = 300
args.nout = args.nhid_glat_l
# ===================
# Baseline Model Parameters
args.Trans_num = 3
args.GAT_num = 3  # increase attention multiple head parallel or in series
args.fea_dim = 300
args.nhid_gat = 300   #statt with 300
args.nhid_trans = 300
args.n_heads = 8
args.batch_size = 200
args.mini_node_num = 40
args.weight_decay = 5e-4
args.lr = 0.0001
# args.lr = 0.00001 * args.batch_size
args.step_size = 15 # tuning this number, maybe change this to adaptive learning rate in the future depending on the loss
args.ratio = 5
args.if_max_length_fix = True

# record_file_name = date + "_{}g_{}t_concat_no_init_mask.txt".format(args.GAT_num, args.Trans_num)

args.struct = [str(num) for num in args.struct]
record_file_name = date + "_" + "_".join(args.struct) + "_concat_no_init_mask.txt"
# args.weight_decay = 1e-4

args.outdir = "log/"
args.model_outdir = "models/"

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

global records
global records_count


records = ""
records_count = 0
fc = False

print("record_file_name: ", record_file_name)

acc_recorder = utils.Record()

def write_to_file(file_name, content):
    fh = open(file_name, "a")
    fh.write(content)
    fh.close
    content = ""
    return content


def save_to_record(content):
    global records
    global records_count

    records += (content + "\n")
    records_count += 1

    file_name = args.outdir + record_file_name

    if records_count % 1 == 0:
        write_to_file(file_name, records)
        records = ""


save_to_record(str(args))


def my_collate(batch):
    max_length = 0

    remove_list = []
    for i, item in enumerate(batch):
        if item[0].size(0) >= args.mini_node_num:
            remove_list.append(i)
        else:
            max_length = max(max_length, item[0].size(0))

    if args.if_max_length_fix:
        max_length = args.mini_node_num
    # print(max_length)
    # max_length = max_length + 1
    gt_embeds = []
    input_embeds = []
    adjs = []
    input_masks = []
    pad_masks = []
    for i, (gt_embed, input_embed, adj, input_mask) in enumerate(batch):
        if i not in remove_list:
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
    # if max_length > 100:
    #     print('max_length:', max_length)
    #     save_to_record("".join(['max_length:', str(max_length)]))
    return [gt_embeds, input_embeds, adjs, input_masks, pad_masks]


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
        num_neg = len(torch.nonzero(neg_mask))
        mask = random.sample(range(0, num_neg), num_neg - args.ratio * num_pos)
        bal_neg_mask[torch.nonzero(neg_mask)[mask]] = 0

    return torch.nonzero(pos_mask).squeeze(-1), torch.nonzero(neg_mask).squeeze(-1), torch.nonzero(bal_neg_mask).squeeze(-1), torch.nonzero(effective_mask).squeeze(-1)

    # torch.cat((pad_masks.repeat(1, N).view(B, N * N), pad_masks.repeat(1, N, 1)), dim=2).view(B, N, -1, 2 * D)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=my_collate)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

test_dataset = VG_data(status='test', data_root=os.path.join(home_path, 'data'))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=my_collate)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)


vocab_num = train_dataset.vocabnum()

# Model and optimizer
# model = Baseline(vocab_num=vocab_num,
#                          feat_dim=args.fea_dim,
#                          nhid_gat=args.nhid_gat,
#                          nhid_trans=args.nhid_trans,
#                          dropout=args.dropout,
#                          nheads=args.n_heads,
#                          alpha=args.alpha,
#                          GAT_num=args.GAT_num,
#                          Trans_num=args.Trans_num,
#                          blank=blank_idx,
#                          fc=fc)

model = GLATNET(vocab_num=vocab_num,
                feat_dim=args.fea_dim,
                nhid_glat_g=args.nhid_glat_g,
                nhid_glat_l=args.nhid_glat_l,
                nout=args.nout,
                dropout=args.dropout,
                nheads=args.n_heads,
                blank=blank_idx,
                types=args.struct)

model = model.to(device=device)
# model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[14, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125], gamma=0.5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

# cri_rec = torch.nn.CrossEntropyLoss()
# cri_rec = torch.nn.NLLLoss(ignore_index=blank_idx)
cri_rec = torch.nn.NLLLoss(ignore_index=blank_idx)
cri_rec = cri_rec.to(device=device)

# cri_con = torch.nn.CrossEntropyLoss()
# cri_con = torch.nn.NLLLoss(ignore_index=blank_idx)
cri_con = torch.nn.NLLLoss()
cri_con = cri_con.to(device=device)


writer = SummaryWriter(log_dir="my_experiment", filename_suffix=record_file_name.split('.')[0])

def train(epoch):
    model.train()
    t = time.time()
    loss_total_rec = 0
    loss_totoal_con = 0
    num_sample = 0
    node_acc = utils.Counter(classes=vocab_num)
    node_acc_mask = utils.Counter(classes=vocab_num)
    edge_acc_train = utils.Counter()
    edge_acc_eff = utils.Counter()

    for i, (gt_embed, input_embed, adj, input_mask, pad_masks) in enumerate(train_loader):

        input_embed = input_embed.to(device=device)
        adj = adj.to(device=device)
        gt_embed = gt_embed.to(device=device)
        pad_masks = pad_masks.to(device=device)
        input_mask = input_mask.to(device=device)

        # keep the last dimension for word length option
        input_embed = input_embed.squeeze(-1)
        gt_embed = gt_embed.squeeze(-1)
        pad_masks = pad_masks.squeeze(-1)
        input_mask = input_mask.squeeze(-1)

        if adj.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:
            pred_label, pred_connect = model(input_embed, adj)
            pred_label_flat = pred_label.view(-1, pred_label.size(-1))
            # gt_embed = gt_embed.squeeze(-1).view(-1)
            gt_embed_flat = gt_embed.view(-1)
            pos_mask, neg_mask, bal_neg_mask, effective_mask = get_gt_edge(pad_masks, adj)
            pred_connect_train = torch.cat((pred_connect[pos_mask], pred_connect[bal_neg_mask]), 0)
            gt_edge_train = torch.cat((torch.ones(len(pos_mask)), torch.zeros(len(bal_neg_mask))), 0).long()

            pred_connect_eff = torch.cat((pred_connect[pos_mask], pred_connect[neg_mask]), 0)
            gt_edge_eff = torch.cat((torch.ones(len(pos_mask)), torch.zeros(len(neg_mask))), 0).long()

            # torch.cat((pred_connect[pos_mask], pred_connect[neg_mask]), 0)
            # torch.cat((torch.ones(num_list[0]), torch.zeros(num_list[1])), 0).long()
            # pdb.set_trace()
            pred_label_eff = pred_label_flat[pad_masks.view(-1) == 0]
            gt_embed_eff = gt_embed_flat[pad_masks.view(-1) == 0]
            pred_label_mask = pred_label_flat[input_mask.view(-1) == 1]
            gt_embed_mask = gt_embed_flat[input_mask.view(-1) == 1]

            # loss_rec = cri_rec(pred_label_eff, gt_embed_eff)
            loss_rec = cri_rec(pred_label_flat, gt_embed_flat)
            loss_con = cri_con(pred_connect_train, gt_edge_train.to(device=device))
            # loss_con = cri_con(pred_connect_eff, gt_edge_eff.to(device=device))
            loss = loss_rec + loss_con
            # loss = loss_rec

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_sample += 1
            loss_total_rec += loss_rec.item()
            loss_totoal_con += loss_con.item()

            node_acc.add(pred_label_eff, gt_embed_eff)
            node_acc_mask.add(pred_label_mask, gt_embed_mask)
            edge_acc_eff.add(pred_connect_eff, gt_edge_eff)
            edge_acc_train.add(pred_connect_train, gt_edge_train)

            # if not args.fastmode:
            #     # Evaluate validation set performance separately,
            #     # deactivates dropout during validation run.
            #     model.eval()
            #     output = model(features, adj)

            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = accuracy(output[idx_val], labels[idx_val])
            if (i+1) % 100 == 0:
                print('Train Epoch: {:04d} [{}/{}] '.format(epoch, i, len(train_loader)),
                      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
                      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
                      'node_acc_train: {:.4f} '.format(node_acc.overall_acc()),
                      'node_acc_mask_train: {:.4f} '.format(node_acc_mask.overall_acc()),
                      'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
                      'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
                      'edge_acc_eff_classacc: {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                                    edge_acc_eff.class_acc()[1]),
                      'edge_acc_eff_recall: {:.4f} '.format(edge_acc_eff.recall()[0]),
                      'time: {:.4f}s '.format(time.time() - t))
                # 'loss_val: {:.4f}'.format(loss_val.item()),
                # 'acc_val: {:.4f}'.format(acc_val.item()),
                save_to_record("".join(['Train Epoch: {:04d} [{}/{}] '.format(epoch, i, len(train_loader)),
                      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
                      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
                      'node_acc_train: {:.4f} '.format(node_acc.overall_acc()),
                      'node_acc_mask_train: {:.4f} '.format(node_acc_mask.overall_acc()),
                      'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
                      'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
                      'edge_acc_eff_classacc: {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                                    edge_acc_eff.class_acc()[1]),
                      'edge_acc_eff_recall: {:.4f} '.format(edge_acc_eff.recall()[0]),
                      'time: {:.4f}s '.format(time.time() - t)]))
                # 'loss_val: {:.4f}'.format(loss_val.item()),
                # 'acc_val: {:.4f}'.format(acc_val.item()),
        # torch.cuda.empty_cache()

    print('Train Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
          'node_acc_train: {:.4f} '.format(node_acc.overall_acc()),
          'node_acc_mask_train: {:.4f} '.format(node_acc_mask.overall_acc()),
          'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
          'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
          'edge_acc_eff_classacc: {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                        edge_acc_eff.class_acc()[1]),
          'edge_acc_eff_recall: {:.4f} '.format(edge_acc_eff.recall()[0]),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t))
    save_to_record("".join(['Train Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
          'node_acc_train: {:.4f} '.format(node_acc.overall_acc()),
          'node_acc_mask_train: {:.4f} '.format(node_acc_mask.overall_acc()),
          'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
          'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
          'edge_acc_eff_classacc: {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                        edge_acc_eff.class_acc()[1]),
          'edge_acc_eff_recall: {:.4f} '.format(edge_acc_eff.recall()[0]),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t)]))
    writer.add_scalar('train/loss_rec', loss_total_rec/num_sample, epoch)
    writer.add_scalar('train/loss_con', loss_totoal_con/num_sample, epoch)
    writer.add_scalar('train/node_acc_train', node_acc.overall_acc(), epoch)
    writer.add_scalar('train/node_acc_mask_train', node_acc_mask.overall_acc(), epoch)
    writer.add_scalar('train/edge_acc_train_overallacc', edge_acc_train.overall_acc(), epoch)
    writer.add_scalar('train/edge_acc_eff_overallacc', edge_acc_eff.overall_acc(), epoch)
    writer.add_scalar('train/edge_acc_eff_classacc_neg', edge_acc_eff.class_acc()[0], epoch)
    writer.add_scalar('train/edge_acc_eff_classacc_pos', edge_acc_eff.class_acc()[1], epoch)
    writer.add_scalar('train/edge_acc_eff_recall', edge_acc_eff.recall()[0], epoch)


def test(epoch):
    model.eval()
    t = time.time()

    loss_total_rec = 0
    loss_totoal_con = 0
    num_sample = 0
    node_acc = utils.Counter(classes=vocab_num)
    node_acc_mask = utils.Counter(classes=vocab_num)
    edge_acc = utils.Counter()

    for i, (gt_embed, input_embed, adj, input_mask, pad_masks) in enumerate(test_loader):
        input_embed = input_embed.to(device=device)
        adj = adj.to(device=device)
        gt_embed = gt_embed.to(device=device)
        pad_masks = pad_masks.to(device=device)
        input_mask = input_mask.to(device=device)

        # input_mask = input_mask.to(device=device)

        input_embed = input_embed.squeeze(-1)
        gt_embed = gt_embed.squeeze(-1)
        pad_masks = pad_masks.squeeze(-1)
        input_mask = input_mask.squeeze(-1)

        if adj.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:
            pred_label, pred_connect = model(input_embed, adj)
            pred_label_flat = pred_label.view(-1, pred_label.size(-1))
            # gt_embed = gt_embed.squeeze(-1).view(-1)
            gt_embed_flat = gt_embed.view(-1)
            pos_mask, neg_mask, bal_neg_mask, effective_mask = get_gt_edge(pad_masks, adj)
            # pdb.set_trace()
            pred_label_mask = pred_label_flat[input_mask.view(-1)==1]
            gt_embed_mask = gt_embed_flat[input_mask.view(-1)==1]

            pred_connect_eff = torch.cat((pred_connect[pos_mask], pred_connect[neg_mask]), 0)
            gt_edge_eff = torch.cat((torch.ones(len(pos_mask)), torch.zeros(len(neg_mask))), 0).long()

            pred_label_eff = pred_label_flat[pad_masks.view(-1)==0]
            gt_embed_eff = gt_embed_flat[pad_masks.view(-1)==0]
            # loss_rec = cri_rec(pred_label_eff, gt_embed_eff)
            loss_rec = cri_rec(pred_label_flat, gt_embed_flat)
            loss_con = cri_con(pred_connect_eff, gt_edge_eff.to(device=device))
            # loss = loss_rec + loss_con

            num_sample += 1
            loss_total_rec += loss_rec.item()
            loss_totoal_con += loss_con.item()

            node_acc_mask.add(pred_label_mask, gt_embed_mask)
            node_acc.add(pred_label_eff, gt_embed_eff)
            edge_acc.add(pred_connect_eff, gt_edge_eff)
        # torch.cuda.empty_cache()

    print('Test Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
          'node_acc_eff: {:.4f} '.format(node_acc.overall_acc()),
          'node_acc_mask_train: {:.4f} '.format(node_acc_mask.overall_acc()),
          'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc.overall_acc()),
          'edge_acc_eff_classacc: {:.4f} {:.4f} '.format(edge_acc.class_acc()[0],
                                                        edge_acc.class_acc()[1]),
          'edge_acc_eff_recall: {:.4f} '.format(edge_acc.recall()[0]),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t))
    save_to_record("".join(['Test Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
          'node_acc_eff: {:.4f} '.format(node_acc.overall_acc()),
          'node_acc_mask_train: {:.4f} '.format(node_acc_mask.overall_acc()),
          'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc.overall_acc()),
          'edge_acc_eff_classacc: {:.4f} {:.4f} '.format(edge_acc.class_acc()[0],
                                                        edge_acc.class_acc()[1]),
          'edge_acc_eff_recall: {:.4f} '.format(edge_acc.recall()[0]),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t)]))
    writer.add_scalar('test/loss_rec', loss_total_rec/num_sample, epoch)
    writer.add_scalar('test/loss_con', loss_totoal_con/num_sample, epoch)
    writer.add_scalar('test/node_acc_train', node_acc.overall_acc(), epoch)
    writer.add_scalar('test/node_acc_mask_train', node_acc_mask.overall_acc(), epoch)
    writer.add_scalar('test/edge_acc_eff_overallacc', edge_acc.overall_acc(), epoch)
    writer.add_scalar('test/edge_acc_eff_classacc_neg', edge_acc.class_acc()[0], epoch)
    writer.add_scalar('test/edge_acc_eff_classacc_pos', edge_acc.class_acc()[1], epoch)
    writer.add_scalar('test/edge_acc_eff_recall', edge_acc.recall()[0], epoch)

    if acc_recorder.compare_node_mask_acc(node_acc_mask.overall_acc()):
        utils.save_model(model, epoch, "best_test_node_mask_acc", args.model_outdir, record_file_name.split(".")[0],
                         acc_recorder.get_best_test_node_mask_acc())
    if acc_recorder.compare_edge_pos_acc(edge_acc.class_acc()[1]):
        utils.save_model(model, epoch, "best_test_edge_pos_acc", args.model_outdir, record_file_name.split(".")[0],
                         acc_recorder.get_best_test_edge_pos_acc())

    print("best node_mask_acc {:.4f}".format(acc_recorder.get_best_test_node_mask_acc()))
    print("best edge_pos_acc {:.4f}".format(acc_recorder.get_best_test_edge_pos_acc()))
    save_to_record("best node_mask_acc {:.4f}".format(acc_recorder.get_best_test_node_mask_acc()))
    save_to_record("best edge_pos_acc {:.4f}".format(acc_recorder.get_best_test_edge_pos_acc()))


# Train model
t_total = time.time()

save_to_record("")

for epoch in range(args.epochs):
    scheduler.step()

    train(epoch)

    # if (epoch+1) % 5 == 0:
    test(epoch)

writer.close()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()


# Todo:
# 1. loss
# 2. dimension matching
# 3. saving & loading vocab info
# 4. (argument of Attention kvq)