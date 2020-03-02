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
from data_kern import VG_data
from torch.utils.data import DataLoader
from models_kern import Baseline, GLATNET
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
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
parser.add_argument('--bidir', type=int, default=0, help='Adj connection needs to be bidirectional')
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')

args = parser.parse_args()

print('gpu used:', args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

args.cuda = not args.no_cuda and torch.cuda.is_available()

# ToDo: hyperparameters connect with command line later
# ===================
# GLAT Model Additional Parameters
# args.struct = [1, 1, 1]
num_global = 0
num_local = 0
num_glat = 6

args.struct = [1]*num_local + [0]*num_global +[2]*num_glat

# args.struct = [0]*num_global + [1]*num_local +[2]*num_glat

args.nhid_glat_g = 300
args.nhid_glat_l = 300
args.nout = args.nhid_glat_l
# ===================
# Baseline Model Parameters
args.model_pretrained_path = ""
args.Trans_num = 3
args.GAT_num = 3  # increase attention multiple head parallel or in series
args.fea_dim = 300
args.nhid_gat = 300   #statt with 300
args.nhid_trans = 300
args.n_heads = 8
# args.batch_size = 50
args.batch_size = 100
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
global records
global records_count
global model

device = 'cuda'

# home_path = os.getcwd()
home_path = '/home/haoxuan/code/KERN/'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
train_dataset = VG_data(status='train', data_root=os.path.join(home_path, 'data'))

blank_idx = train_dataset.get_blank()

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
    gt_classes = []
    input_classes = []
    adjs = []
    input_masks = []
    # pad_masks = []
    node_types = []

    for i, (gt_class, input_class, adj, input_mask, node_type) in enumerate(batch):
        if i not in remove_list:
            # print(i)
            gt_classes.append(torch.cat((gt_class, blank_idx*torch.ones(max_length-gt_class.size(0), dtype=torch.long)), 0).unsqueeze(0))
            input_classes.append(torch.cat((input_class, blank_idx*torch.ones(max_length-input_class.size(0), dtype=torch.long)), 0).unsqueeze(0))
            # pdb.set_trace()
            new_adj = torch.cat((adj, torch.zeros(max_length-adj.size(0), adj.size(1), dtype=torch.float)), 0)
            new_adj = torch.cat((new_adj, torch.zeros(new_adj.size(0), max_length-new_adj.size(1), dtype=torch.float)), 1)
            # pdb.set_trace()
            adjs.append(new_adj.unsqueeze(0))
            # pdb.set_trace()
            input_masks.append(torch.cat((input_mask, torch.zeros(max_length-input_mask.size(0), dtype=torch.long)), 0).unsqueeze(0))
            # pdb.set_trace()
            # pad_masks.append(torch.cat((torch.zeros_like(gt_class), torch.ones(max_length-gt_class.size(0), dtype=torch.long)), 0).unsqueeze(0))
            node_types.append(torch.cat((node_type, 2 * torch.ones(max_length-node_type.size(0), dtype=torch.long)), 0).unsqueeze(0))
            # node types 0:predicate 1:entity 2:blank/padding

    gt_classes = torch.cat(gt_classes, 0)
    input_classes = torch.cat(input_classes, 0)
    adjs = torch.cat(adjs, 0)
    adjs_lbl = adjs
    adjs_con = torch.clamp(adjs, 0, 1)
    input_masks = torch.cat(input_masks, 0)
    node_types = torch.cat(node_types, 0)
    # pad_masks = torch.cat(pad_masks, 0)
    # if max_length > 100:
    #     print('max_length:', max_length)
    #     save_to_record("".join(['max_length:', str(max_length)]))
    return [gt_classes, input_classes, adjs_con, adjs_lbl, input_masks, node_types]


def get_edge_info(pad_masks, adjs_connect, adjs_label):
    '''

    :param pad_masks: (B, N)
    :param adjs_connect:  (B, N, N) 0-nocon 1-con
    :param adjs_label:  (B, N, N) 0-nocon 1-sub2pred, 2-pred2obj
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
    # pos_mask = adj.view(B, N*N).view(-1)
    pos_mask_label = adjs_label.view(B, N*N).view(-1)
    pos_mask_label_1 = torch.nonzero(torch.eq(pos_mask_label, 1)).squeeze(-1)
    pos_mask_label_2 = torch.nonzero(torch.eq(pos_mask_label, 2)).squeeze(-1)
    pos_mask = adjs_connect.view(B, N*N).view(-1)
    neg_mask = (1-pos_mask.clamp(0, 1)).long()

    neg_mask = torch.mul(neg_mask, effective_mask)
    bal_neg_mask = copy.deepcopy(neg_mask)
    if len(torch.nonzero(neg_mask)) > len(torch.nonzero(pos_mask)):
        num_pos = len(torch.nonzero(pos_mask))
        num_neg = len(torch.nonzero(neg_mask))
        mask = random.sample(range(0, num_neg), num_neg - args.ratio * num_pos)
        bal_neg_mask[torch.nonzero(neg_mask)[mask]] = 0

    return torch.nonzero(pos_mask).squeeze(-1), pos_mask_label_1,  pos_mask_label_2, torch.nonzero(neg_mask).squeeze(-1), torch.nonzero(bal_neg_mask).squeeze(-1), torch.nonzero(effective_mask).squeeze(-1)

    # torch.cat((pad_masks.repeat(1, N).view(B, N * N), pad_masks.repeat(1, N, 1)), dim=2).view(B, N, -1, 2 * D)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=my_collate)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

test_dataset = VG_data(status='test', data_root=os.path.join(home_path, 'data'))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=my_collate)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)


# vocab_num = train_dataset.vocabnum()
vocab_num = train_dataset.vocab_num()
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
    node_acc_entity = utils.Counter(classes=vocab_num[1])
    node_acc_predicate = utils.Counter(classes=vocab_num[0])
    node_acc_mask_entity = utils.Counter(classes=vocab_num[1])
    node_acc_mask_predicate = utils.Counter(classes=vocab_num[0])
    edge_acc_train = utils.Counter(classes=3)
    edge_acc_eff = utils.Counter(classes=3)

    for i, (gt_class, input_class, adj_con, adj_lbl, input_mask, node_type) in enumerate(train_loader):

        input_class = input_class.to(device=device)
        adj_con = adj_con.to(device=device)
        adj_lbl = adj_lbl.to(device=device)
        # adj = adj.to(device=device)
        gt_class = gt_class.to(device=device)
        node_type = node_type.to(device=device)
        input_mask = input_mask.to(device=device)

        # keep the last dimension for word length option
        # input_embed = input_embed.squeeze(-1)
        # gt_embed = gt_embed.squeeze(-1)
        # pad_masks = pad_masks.squeeze(-1)
        # input_mask = input_mask.squeeze(-1)

        if adj_con.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:
            # assert ((args.bidir != False) or any(adj_con[0][0,:] != adj_con[0][:,0]))
            # pdb.set_trace()

            if args.bidir:
                # print('make matrix symmetric')
                adj_con_per = adj_con.permute(0, 2, 1)
                adj_con = adj_con + adj_con_per

            if args.bidir:
                adj_lbl_per = adj_lbl.permute(0, 2, 1)
                adj_lbl = adj_lbl + adj_lbl_per

            # pdb.set_trace()

            pred_label, pred_connect = model(input_class, adj_con, node_type)
            pred_label_predicate = pred_label[0]  # flatten predicate
            pred_label_entities = pred_label[1]  # flatten entities
            pred_label_all = pred_label[2]  # unflatten all labels
            pred_label_flat = pred_label_all.view(-1)

            # gt_embed = gt_embed.squeeze(-1).view(-1)
            gt_class_flat = gt_class.view(-1)
            node_type_flat = node_type.view(-1)
            pos_mask, pos_mask_1, pos_mask_2, neg_mask, bal_neg_mask, effective_mask = get_edge_info((node_type==2).long(), adj_con, adj_lbl)
            pred_connect_train = torch.cat((pred_connect[pos_mask_1], pred_connect[pos_mask_2], pred_connect[bal_neg_mask]), 0)
            gt_edge_train = torch.cat((torch.ones(len(pos_mask_1)), 2 * torch.ones(len(pos_mask_2)), torch.zeros(len(bal_neg_mask))), 0).long()

            pred_connect_eff = torch.cat((pred_connect[pos_mask_1], pred_connect[pos_mask_2], pred_connect[neg_mask]), 0)
            gt_edge_eff = torch.cat((torch.ones(len(pos_mask_1)), 2 * torch.ones(len(pos_mask_2)), torch.zeros(len(neg_mask))), 0).long()

            pred_label_eff_entity = pred_label_flat[node_type.view(-1) == 1]
            gt_class_eff_entity = gt_class_flat[node_type.view(-1) == 1]
            pred_label_eff_predicate = pred_label_flat[node_type.view(-1) == 0]
            gt_class_eff_predicate = gt_class_flat[node_type.view(-1) == 0]

            # pred_label_mask_entity = pred_label_flat[(input_mask.view(-1) == 1) * (node_type.view(-1) == 1)]
            # gt_class_mask_entity = gt_class_flat[(input_mask.view(-1) == 1) * (node_type.view(-1) == 1)]
            # pdb.set_trace()
            pred_label_mask_predicate= pred_label_flat[(input_mask.view(-1) == 1) * (node_type.view(-1) == 0)]
            gt_class_mask_predicate = gt_class_flat[(input_mask.view(-1) == 1) * (node_type.view(-1) == 0)]

            # loss_rec = cri_rec(pred_label_eff, gt_embed_eff)
            loss_rec_predicate = cri_rec(pred_label_predicate, gt_class_flat[node_type_flat==0])
            loss_rec_entity = cri_rec(pred_label_entities, gt_class_flat[node_type_flat==1])
            loss_rec = loss_rec_entity + loss_rec_predicate
            # pdb.set_trace()
            # loss_con = cri_con(pred_connect_train, gt_edge_train.to(device=device))
            # loss_con = cri_con(pred_connect_eff, gt_edge_eff.to(device=device))
            # loss = loss_rec + loss_con
            loss = loss_rec

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_sample += 1
            loss_total_rec += loss_rec.item()
            # loss_totoal_con += loss_con.item()

            node_acc_entity.add(pred_label_eff_entity, gt_class_eff_entity)
            node_acc_predicate.add(pred_label_eff_predicate, gt_class_eff_predicate)
            # node_acc_mask_entity.add(pred_label_mask_entity, gt_class_mask_entity)
            node_acc_mask_predicate.add(pred_label_mask_predicate, gt_class_mask_predicate)
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
                      'node_acc_entity_train: {:.4f} '.format(node_acc_entity.overall_acc()),
                      'node_acc_predicate_train: {:.4f} '.format(node_acc_predicate.overall_acc()),
                      # 'node_acc_mask_entity_train: {:.4f} '.format(node_acc_mask_entity.overall_acc()),
                      'node_acc_mask_predicate_train: {:.4f} '.format(node_acc_mask_predicate.overall_acc()),
                      'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
                      'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
                      'edge_acc_eff_classacc: {:.4f} {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                                    edge_acc_eff.class_acc()[1],
                                                                     edge_acc_eff.class_acc()[2]),
                      'edge_acc_eff_recall: {:.4f} {:.4f}'.format(edge_acc_eff.recall()[1], edge_acc_eff.recall()[2]),
                      'time: {:.4f}s '.format(time.time() - t))
                # 'loss_val: {:.4f}'.format(loss_val.item()),
                # 'acc_val: {:.4f}'.format(acc_val.item()),
                save_to_record("".join(['Train Epoch: {:04d} [{}/{}] '.format(epoch, i, len(train_loader)),
                      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
                      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
                      # 'node_acc_train: {:.4f} '.format(node_acc.overall_acc()),
                      # 'node_acc_mask_train: {:.4f} '.format(node_acc_mask.overall_acc()),
                      'node_acc_entity_train: {:.4f} '.format(node_acc_entity.overall_acc()),
                      'node_acc_predicate_train: {:.4f} '.format(node_acc_predicate.overall_acc()),
                      # 'node_acc_mask_entity_train: {:.4f} '.format(node_acc_mask_entity.overall_acc()),
                      'node_acc_mask_predicate_train: {:.4f} '.format(node_acc_mask_predicate.overall_acc()),
                      'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
                      'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
                      'edge_acc_eff_classacc: {:.4f} {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                                    edge_acc_eff.class_acc()[1],
                                                                     edge_acc_eff.class_acc()[2]),
                      'edge_acc_eff_recall: {:.4f} {:.4f}'.format(edge_acc_eff.recall()[1], edge_acc_eff.recall()[2]),
                      'time: {:.4f}s '.format(time.time() - t)]))
                # 'loss_val: {:.4f}'.format(loss_val.item()),
                # 'acc_val: {:.4f}'.format(acc_val.item()),
        # torch.cuda.empty_cache()

    print('Train Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
          'node_acc_entity_train: {:.4f} '.format(node_acc_entity.overall_acc()),
          'node_acc_predicate_train: {:.4f} '.format(node_acc_predicate.overall_acc()),
          # 'node_acc_mask_entity_train: {:.4f} '.format(node_acc_mask_entity.overall_acc()),
          'node_acc_mask_predicate_train: {:.4f} '.format(node_acc_mask_predicate.overall_acc()),
          'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
          'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
          'edge_acc_eff_classacc: {:.4f} {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                        edge_acc_eff.class_acc()[1],
                                                         edge_acc_eff.class_acc()[2]),
          'edge_acc_eff_recall: {:.4f} {:.4f}'.format(edge_acc_eff.recall()[1], edge_acc_eff.recall()[2]),
          # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t))
    save_to_record("".join(['Train Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
        'node_acc_entity_train: {:.4f} '.format(node_acc_entity.overall_acc()),
        'node_acc_predicate_train: {:.4f} '.format(node_acc_predicate.overall_acc()),
        # 'node_acc_mask_entity_train: {:.4f} '.format(node_acc_mask_entity.overall_acc()),
        'node_acc_mask_predicate_train: {:.4f} '.format(node_acc_mask_predicate.overall_acc()),
          'edge_acc_train_overallacc: {:.4f} '.format(edge_acc_train.overall_acc()),
          'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc_eff.overall_acc()),
          'edge_acc_eff_classacc: {:.4f} {:.4f} {:.4f} '.format(edge_acc_eff.class_acc()[0],
                                                        edge_acc_eff.class_acc()[1],
                                                         edge_acc_eff.class_acc()[2]),
          'edge_acc_eff_recall: {:.4f} {:.4f}'.format(edge_acc_eff.recall()[1],
                                                    edge_acc_eff.recall()[2]),
                            # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t)]))
    writer.add_scalar('train/loss_rec', loss_total_rec/num_sample, epoch)
    writer.add_scalar('train/loss_con', loss_totoal_con/num_sample, epoch)
    writer.add_scalar('train/node_acc_train_entity', node_acc_entity.overall_acc(), epoch)
    writer.add_scalar('train/node_acc_train_predicate', node_acc_predicate.overall_acc(), epoch)
    # writer.add_scalar('train/node_acc_mask_train_entity', node_acc_mask_entity.overall_acc(), epoch)
    writer.add_scalar('train/node_acc_mask_train_predicate', node_acc_mask_predicate.overall_acc(), epoch)
    writer.add_scalar('train/edge_acc_train_overallacc', edge_acc_train.overall_acc(), epoch)
    writer.add_scalar('train/edge_acc_eff_overallacc', edge_acc_eff.overall_acc(), epoch)
    writer.add_scalar('train/edge_acc_eff_classacc_neg', edge_acc_eff.class_acc()[0], epoch)
    writer.add_scalar('train/edge_acc_eff_classacc_pos1', edge_acc_eff.class_acc()[1], epoch)
    writer.add_scalar('train/edge_acc_eff_classacc_pos2', edge_acc_eff.class_acc()[2], epoch)
    writer.add_scalar('train/edge_acc_eff_recall1', edge_acc_eff.recall()[1], epoch)
    writer.add_scalar('train/edge_acc_eff_recall2', edge_acc_eff.recall()[2], epoch)


def test(epoch):
    model.eval()
    t = time.time()

    loss_total_rec = 0
    loss_totoal_con = 0
    num_sample = 0
    node_acc_entity = utils.Counter(classes=vocab_num[1])
    node_acc_predicate = utils.Counter(classes=vocab_num[0])
    node_acc_mask_predicate = utils.Counter(classes=vocab_num[0])
    # node_acc = utils.Counter(classes=vocab_num[0])
    # node_acc_mask = utils.Counter(classes=vocab_num[0])
    edge_acc = utils.Counter(classes=3)

    for i, (gt_class, input_class, adj_con, adj_lbl, input_mask, node_type) in enumerate(test_loader):
        input_class = input_class.to(device=device)
        adj_con = adj_con.to(device=device)
        adj_lbl = adj_lbl.to(device=device)
        # adj = adj.to(device=device)
        gt_class = gt_class.to(device=device)
        node_type = node_type.to(device=device)
        input_mask = input_mask.to(device=device)
        # input_mask = input_mask.to(device=device)

        # input_embed = input_embed.squeeze(-1)
        # gt_embed = gt_embed.squeeze(-1)
        # pad_masks = pad_masks.squeeze(-1)
        # input_mask = input_mask.squeeze(-1)

        if adj_con.size(1) == 1:
            print('{} skip for 1 node'.format(i))
            continue
        else:

            if args.bidir:
                adj_con_per = adj_con.permute(0, 2, 1)
                adj_con = adj_con + adj_con_per

            if args.bidir:
                adj_lbl_per = adj_lbl.permute(0, 2, 1)
                adj_lbl = adj_lbl + adj_lbl_per

            pred_label, pred_connect = model(input_class, adj_con, node_type)
            pred_label_predicate = pred_label[0]  # flatten predicate
            pred_label_entities = pred_label[1]  # flatten entities
            pred_label_all = pred_label[2]  # unflatten all labels
            pred_label_flat = pred_label_all.view(-1)


            # pred_label_flat = pred_label.view(-1, pred_label.size(-1))
            # gt_embed = gt_embed.squeeze(-1).view(-1)
            gt_class_flat = gt_class.view(-1)
            node_type_flat = node_type.view(-1)

            pos_mask, pos_mask_1, pos_mask_2, neg_mask, bal_neg_mask, effective_mask = get_edge_info((node_type==2).long(), adj_con, adj_lbl)

            # pred_label_mask = pred_label_flat[input_mask.view(-1)==1]
            # gt_embed_mask = gt_embed_flat[input_mask.view(-1)==1]
            pred_label_mask_predicate= pred_label_flat[(input_mask.view(-1) == 1) * (node_type.view(-1) == 0)]
            gt_class_mask_predicate = gt_class_flat[(input_mask.view(-1) == 1) * (node_type.view(-1) == 0)]

            # pred_connect_eff = torch.cat((pred_connect[pos_mask_1], pred_connect[pos_mask_2], pred_connect[neg_mask]), 0)
            # gt_edge_eff = torch.cat((torch.ones(len(pos_mask_1)), 2 * torch.ones(len(pos_mask_2)), torch.zeros(len(neg_mask))), 0).long()
            pred_connect_eff = torch.cat((pred_connect[pos_mask_1], pred_connect[pos_mask_2], pred_connect[neg_mask]), 0)
            gt_edge_eff = torch.cat((torch.ones(len(pos_mask_1)), 2 * torch.ones(len(pos_mask_2)), torch.zeros(len(neg_mask))), 0).long()


            # pred_connect_eff = torch.cat((pred_connect[pos_mask], pred_connect[neg_mask]), 0)
            # gt_edge_eff = torch.cat((torch.ones(len(pos_mask)), torch.zeros(len(neg_mask))), 0).long()

            # pred_label_eff = pred_label_flat[pad_masks.view(-1)==0]
            # gt_embed_eff = gt_embed_flat[pad_masks.view(-1)==0]
            pred_label_eff_entity = pred_label_flat[node_type.view(-1) == 1]
            gt_class_eff_entity = gt_class_flat[node_type.view(-1) == 1]
            pred_label_eff_predicate = pred_label_flat[node_type.view(-1) == 0]
            gt_class_eff_predicate = gt_class_flat[node_type.view(-1) == 0]

            # loss_rec = cri_rec(pred_label_eff, gt_embed_eff)
            # loss_rec = cri_rec(pred_label_flat, gt_embed_flat)
            loss_rec_predicate = cri_rec(pred_label_predicate, gt_class_flat[node_type_flat==0])
            loss_rec_entity = cri_rec(pred_label_entities, gt_class_flat[node_type_flat==1])
            loss_rec = loss_rec_entity + loss_rec_predicate
            # loss_con = cri_con(pred_connect_eff, gt_edge_eff.to(device=device))
            # loss = loss_rec + loss_con

            num_sample += 1
            loss_total_rec += loss_rec.item()
            # loss_totoal_con += loss_con.item()

            # node_acc_mask.add(pred_label_mask, gt_embed_mask)
            # node_acc.add(pred_label_eff, gt_embed_eff)
            # edge_acc.add(pred_connect_eff, gt_edge_eff)

            node_acc_entity.add(pred_label_eff_entity, gt_class_eff_entity)
            node_acc_predicate.add(pred_label_eff_predicate, gt_class_eff_predicate)
            node_acc_mask_predicate.add(pred_label_mask_predicate, gt_class_mask_predicate)
            edge_acc.add(pred_connect_eff, gt_edge_eff)

        # torch.cuda.empty_cache()

    print('Test Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
          'node_acc_entity_train: {:.4f} '.format(node_acc_entity.overall_acc()),
          'node_acc_predicate_train: {:.4f} '.format(node_acc_predicate.overall_acc()),
          # 'node_acc_mask_entity_train: {:.4f} '.format(node_acc_mask_entity.overall_acc()),
          'node_acc_mask_predicate_train: {:.4f} '.format(node_acc_mask_predicate.overall_acc()),
          'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc.overall_acc()),
          'edge_acc_eff_classacc: {:.4f} {:.4f} {:.4f} '.format(edge_acc.class_acc()[0],
                                                                edge_acc.class_acc()[1],
                                                                edge_acc.class_acc()[2]),
          'edge_acc_eff_recall: {:.4f} {:4f}'.format(edge_acc.recall()[1], edge_acc.recall()[2]),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t))
    save_to_record("".join(['Test Epoch Finished: {:04d} '.format(epoch),
      'loss_rec: {:.4f} '.format(loss_total_rec/num_sample),
      'loss_con: {:.4f} '.format(loss_totoal_con/num_sample),
      'node_acc_entity_train: {:.4f} '.format(node_acc_entity.overall_acc()),
      'node_acc_predicate_train: {:.4f} '.format(node_acc_predicate.overall_acc()),
      'node_acc_mask_predicate_train: {:.4f} '.format(node_acc_mask_predicate.overall_acc()),
      'edge_acc_eff_overallacc: {:.4f} '.format(edge_acc.overall_acc()),
      'edge_acc_eff_classacc: {:.4f} {:.4f} {:.4f} '.format(edge_acc.class_acc()[0],
                                                            edge_acc.class_acc()[1],
                                                            edge_acc.class_acc()[2]),
      'edge_acc_eff_recall: {:.4f} {:.4f}'.format(edge_acc.recall()[1], edge_acc.recall()[2]),
      # 'acc_train: {:.4f}'.format(acc_train.item()),
      # 'loss_val: {:.4f}'.format(loss_val.item()),
      # 'acc_val: {:.4f}'.format(acc_val.item()),
      'time: {:.4f}s '.format(time.time() - t)]))
    writer.add_scalar('test/loss_rec', loss_total_rec/num_sample, epoch)
    writer.add_scalar('test/loss_con', loss_totoal_con/num_sample, epoch)
    writer.add_scalar('train/node_acc_train_entity', node_acc_entity.overall_acc(), epoch)
    writer.add_scalar('train/node_acc_train_predicate', node_acc_predicate.overall_acc(), epoch)
    # writer.add_scalar('train/node_acc_mask_train_entity', node_acc_mask_entity.overall_acc(), epoch)
    writer.add_scalar('train/node_acc_mask_train_predicate', node_acc_mask_predicate.overall_acc(), epoch)
    writer.add_scalar('test/edge_acc_eff_overallacc', edge_acc.overall_acc(), epoch)
    writer.add_scalar('test/edge_acc_eff_classacc_neg', edge_acc.class_acc()[0], epoch)
    writer.add_scalar('train/edge_acc_eff_classacc_pos1', edge_acc.class_acc()[1], epoch)
    writer.add_scalar('train/edge_acc_eff_classacc_pos2', edge_acc.class_acc()[2], epoch)
    writer.add_scalar('train/edge_acc_eff_recall1', edge_acc.recall()[1], epoch)
    writer.add_scalar('train/edge_acc_eff_recall2', edge_acc.recall()[2], epoch)

    if acc_recorder.compare_node_mask_acc(node_acc_mask_predicate.overall_acc()):
        utils.save_model(model, epoch, "best_test_node_mask_predicate_acc", args.model_outdir, record_file_name.split(".")[0],
                         acc_recorder.get_best_test_node_mask_acc())
    if acc_recorder.compare_edge_pos_acc(edge_acc.class_acc()[1]):
        utils.save_model(model, epoch, "best_test_edge_pos_acc", args.model_outdir, record_file_name.split(".")[0],
                         acc_recorder.get_best_test_edge_pos_acc())

    print("best node_mask_acc {:.4f}".format(acc_recorder.get_best_test_node_mask_acc()))
    print("best edge_pos_acc {:.4f}".format(acc_recorder.get_best_test_edge_pos_acc()))
    save_to_record("best node_mask_acc {:.4f}".format(acc_recorder.get_best_test_node_mask_acc()))
    save_to_record("best edge_pos_acc {:.4f}".format(acc_recorder.get_best_test_edge_pos_acc()))


t_total = time.time()


for epoch in range(args.epochs):
    scheduler.step()

    train(epoch)
    # if (epoch+1) % 5 == 0:
    test(epoch)

writer.close()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


