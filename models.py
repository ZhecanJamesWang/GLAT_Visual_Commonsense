import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, Connect_Cls, EncoderLayer, GraphAttentionLayer
import torch

#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid1, nhid2, nhid3, n_class, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid1)
#         self.gc2 = GraphConvolution(nhid1, nhid2)
#         self.gc_reconst = GraphConvolution(nhid2, nfeat)
#         self.gc_connect = Connect_Cls(nhid2, nhid3, n_class)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         # x = x.squeeze()
#         # pdb.set_trace()
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x_reconst = self.gc_reconst(x, adj)
#         x_connect, num_list = self.gc_connect(x, adj)
#
#         return x_reconst, F.log_softmax(x_connect, dim=1), num_list

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, noutput, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        x = F.elu(x)
        return x


class GAT_Ensemble(nn.Module):
    def __init__(self, vocab_num, nfeat, nhid, noutput, dropout, alpha, nheads, GAT_num):
        """Dense version of GAT."""
        super(GAT_Ensemble, self).__init__()
        print("initialize GAT with module num = : ", GAT_num)

        self.GAT_num = GAT_num
        self.GATs = nn.ModuleList()
        self.embed = nn.Embedding(vocab_num, nfeat)

        for num in range(self.GAT_num):
            model = GAT(nfeat, nhid, noutput, dropout, alpha, nheads)
            self.GATs.append(model)

    def forward(self, fea, adj):
        x = self.embed(fea)
        for num in range(self.GAT_num):
            x = self.GATs[num](x, adj)
        return x


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.position_enc = 0

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):


        super().__init__()

        self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k,
                               d_v=d_v, dropout=dropout)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq):
        src_pos= src_seq
        enc_output, *_ = self.encoder(src_seq, src_pos)

        return enc_output


class Transformer_Ensemble(nn.Module):
    def __init__(self, Trans_num, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64,
                 dropout=0.1):
        """Ensembled version of Trans."""
        super(Transformer_Ensemble, self).__init__()
        print("initialize Trans with module num = : ", Trans_num)

        self.Trans_num = Trans_num
        self.Trans = nn.ModuleList()

        for num in range(self.Trans_num):
            model = Transformer(d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=n_layers,
                                n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
            self.Trans.append(model)

    def forward(self, x):
        for num in range(self.Trans_num):
            x = self.Trans[num](x)
        return x


class Pred_label(nn.Module):
    def __init__(self, model):
        super(Pred_label, self).__init__()
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight

    def forward(self, h):
        lm_logits = self.decoder(h)
        return F.softmax(lm_logits, dim=-1)


class Ensemble_encoder(nn.Module):
    def __init__(self, vocab_num, GAT_num, Trans_num, feat_dim, nhid_gat, nhid_trans, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(Ensemble_encoder, self).__init__()
        print("initialize Encoder with GAT num ", GAT_num, " Bert num ", Trans_num)

        self.GAT_Ensemble = GAT_Ensemble(vocab_num, feat_dim, nhid_gat, nhid_trans, dropout, alpha, nheads, GAT_num)

        self.Trans_Ensemble = Transformer_Ensemble(Trans_num, d_word_vec=nhid_trans, d_model=nhid_trans)

        self.Pred_label = Pred_label(self.GAT_Ensemble)
        self.Pred_connect = Connect_Cls(nhid_trans, nhid_trans/2, 2)

    def forward(self, fea, adj):
        x = self.GAT_Ensemble(fea, adj)

        x = self.Trans_Ensemble(x)

        pred_label = self.Pred_label(x)
        pred_edge, num_list = self.Pred_connect(x)

        return pred_label, pred_edge, num_list

