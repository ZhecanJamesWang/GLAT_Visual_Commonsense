import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, Connect_Cls
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
import Constants as Constants
from Layers_transformer import EncoderLayer

import pdb

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        x = F.elu(x)
        return x


class GAT_Ensemble(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads, GAT_num):
        """Dense version of GAT."""
        super(GAT_Ensemble, self).__init__()
        print("initialize GAT with module num = : ", GAT_num)

        self.GAT_num = GAT_num
        self.GATs = nn.ModuleList()

        for num in range(self.GAT_num):
            model = GAT(nfeat, nhid, noutput, dropout, alpha, nheads)
            self.GATs.append(model)

    def forward(self, fea, adj):
        x = fea
        for num in range(self.GAT_num):
            x = self.GATs[num](x, adj)
        return fea, adj, F.log_softmax(x, dim=1)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.

    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        # self.src_word_emb = nn.Embedding(
        #     n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = 0

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_output = src_seq

        enc_output = torch.unsqueeze(enc_output, 0)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            # enc_output, enc_slf_attn = enc_layer(
            #     enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):


        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)


        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq):
        src_pos= src_seq
        enc_output, *_ = self.encoder(src_seq, src_pos)

        return enc_output


class Ensemble_encoder(nn.Module):
    def __init__(self, GAT_num, Bert_num, nfeat, nhid, nclass, dropout, alpha, nheads, total_num):
        """Dense version of GAT."""
        super(Ensemble_encoder, self).__init__()
        print("initialize Encoder with GAT num ", GAT_num, " Bert num ", Bert_num)

        # nclass = 768
        nclass = 10
        self.GAT_Ensemble = GAT_Ensemble(nfeat, nhid, nclass, dropout, alpha, nheads, GAT_num)

        # self.bert = BERT(total_num, hidden=nclass)
        self.transformer = Transformer(total_num, 10, d_word_vec=10, d_model=10)

    def forward(self, fea, adj):
        fea, adj, x = self.GAT_Ensemble(fea, adj)

        # print("bert input: ", x.shape)
        # x = self.bert(x, x)
        x = self.transformer(x)

        return fea, adj, F.log_softmax(x.squeeze(0), dim=1)