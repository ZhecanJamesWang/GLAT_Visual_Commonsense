import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, Connect_Cls
import pdb


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, n_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc_reconst = GraphConvolution(nhid2, nfeat)
        self.gc_connect = Connect_Cls(nhid2, nhid3, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        # x = x.squeeze()
        # pdb.set_trace()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x_reconst = self.gc_reconst(x, adj)
        x_connect, num_list = self.gc_connect(x, adj)

        return x_reconst, F.log_softmax(x_connect, dim=1), num_list
