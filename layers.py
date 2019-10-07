import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import pdb


class Connect_Cls(Module):
    def __init__(self, in_features, mid_features, n_class, bias=True):
        super(Connect_Cls, self).__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.n_class = n_class
        # self.balanced_ratio = 0.5
        self.FC1 = nn.Sequential(
            nn.Linear(2*self.in_features, self.mid_features),
            nn.BatchNorm1d(self.mid_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.FC2 = nn.Linear(self.mid_features, self.n_class)

    def forward(self, input, adj):
        conn = torch.nonzero(adj)
        pos_input = []
        conn_num = len(conn)
        if conn_num != 0:
            for i in range(conn_num):
                pos_input.append(torch.cat([input[conn[i][0]], input[conn[i][1]]], dim=-1).unsqueeze(0))
            pos_input = torch.cat(pos_input, dim=0)
        else:
            pos_input = torch.zeros(size=(0,0), device='cuda')

        neg_input = []
        disconn = torch.nonzero(1-adj)
        if len(conn) <= len(disconn):
            disconn_num = len(conn) if len(conn) != 0 else 2
        else:
            disconn_num = len(disconn)
        for i in range(disconn_num):
            neg_input.append(torch.cat([input[disconn[i][0]], input[disconn[i][1]]], dim=-1).unsqueeze(0))
        neg_input = torch.cat(neg_input, dim=0)

        total_input = torch.cat((pos_input, neg_input), dim=0)
        x = self.FC1(total_input)
        x = self.FC2(x)
        return x, [conn_num, disconn_num]





class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # pdb.set_trace()
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
