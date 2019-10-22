import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, EncoderLayer, GraphAttentionLayer
import torch
# import Constants
import pdb

def get_non_pad_mask(seq, blank):
    assert seq.dim() == 2
    return seq.ne(blank).unsqueeze(-1).type(torch.float)


def get_attn_key_pad_mask(seq_k, seq_q, blank):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.

    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(blank)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, noutput, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, non_pad_mask):

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)

        x *= non_pad_mask

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)

        x *= non_pad_mask

        x = F.elu(x)


        return x


class GAT_Ensemble(nn.Module):
    def __init__(self, vocab_num, nfeat, nhid, noutput, dropout, alpha, nheads, GAT_num):
        """Dense version of GAT."""
        super(GAT_Ensemble, self).__init__()
        print("initialize GAT with module num = : ", GAT_num)

        self.GAT_num = GAT_num
        self.GATs = nn.ModuleList()
        # print(type(vocab_num))
        # print(type(nfeat))
        self.embed = nn.Embedding(vocab_num, nfeat)

        # for num in range(self.GAT_num):
        model = GAT(nfeat, nhid, noutput, dropout, alpha, nheads)
        self.GATs.append(model)

    def forward(self, fea, adj, non_pad_mask):

        fea = fea.long()
        # pdb.set_trace()

        x = self.embed(fea)
        # x = x.squeeze(2)

        # for num in range(self.GAT_num):
        num = 0
        x = self.GATs[num](x, adj, non_pad_mask)
        return x


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):

        super().__init__()

        # n_position = len_max_seq + 1

        # self.src_word_emb = nn.Embedding(
        #     n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        # self.position_enc = 0

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, slf_attn_mask, non_pad_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        # slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        # non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_output = src_seq
        # pdb.set_trace()
        # enc_output = torch.unsqueeze(enc_output, 0)

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
            self, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):


        super().__init__()

        self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k,
                               d_v=d_v, dropout=dropout)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, slf_attn_mask, non_pad_mask):
        src_pos = src_seq
        enc_output, *_ = self.encoder(src_seq, src_pos, slf_attn_mask, non_pad_mask)

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

    def forward(self, x, slf_attn_mask, non_pad_mask):
        for num in range(self.Trans_num):
            x = self.Trans[num](x, slf_attn_mask, non_pad_mask)
        return x


class Connect_Cls(nn.Module):
    def __init__(self, in_features, mid_features, n_class, bias=True):
        super(Connect_Cls, self).__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.n_class = n_class
        # self.balanced_ratio = 0.5
        self.FC = nn.Sequential(
            nn.Linear(2*self.in_features, self.mid_features),
            nn.BatchNorm1d(self.mid_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.mid_features, self.n_class)
        )

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, adj):
        B = input.size(0)
        N = input.size(1)
        D = input.size(2)

        conn_fea = torch.cat([input.repeat(1, 1, N).view(B, N * N, -1), input.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * D) # (B, N, N, 2D)
        conn_fea = conn_fea.view(B, -1, 2*D).view(-1, 2*D)
        conn_adj = adj.view(B, -1).view(-1)

        # pos_conn = torch.nonzero(conn_adj).squeeze(-1)
        # neg_conn = torch.nonzero(1-conn_adj).squeeze(-1)[:len(pos_conn)]
        #
        # pos_fea = conn_fea[pos_conn]
        # neg_fea = conn_fea[neg_conn]
        # total_fea = torch.cat([pos_fea, neg_fea], dim=0)

        # pdb.set_trace()

        x = self.FC(conn_fea)
        # x = self.FC(total_fea)

        # conn = torch.nonzero(adj)
        # pos_input = []
        # conn_num = len(conn)
        # if conn_num != 0:
        #     for i in range(conn_num):
        #         pos_input.append(torch.cat([input[conn[i][0]], input[conn[i][1]]], dim=-1).unsqueeze(0))
        #     pos_input = torch.cat(pos_input, dim=0)
        # else:
        #     pos_input = torch.zeros(size=(0,0), device='cuda')
        #
        # neg_input = []
        # disconn = torch.nonzero(1-adj)
        # if len(conn) <= len(disconn):
        #     disconn_num = len(conn) if len(conn) != 0 else 2
        # else:
        #     disconn_num = len(disconn)
        # for i in range(disconn_num):
        #     neg_input.append(torch.cat([input[disconn[i][0]], input[disconn[i][1]]], dim=-1).unsqueeze(0))
        # neg_input = torch.cat(neg_input, dim=0)
        #
        # total_input = torch.cat((pos_input, neg_input), dim=0)
        # x = self.FC1(total_input)
        # x = self.FC2(x)
        # return x, [len(pos_conn), len(neg_conn)]

        x = self.softmax(x)

        return x


class Pred_label(nn.Module):
    def __init__(self, model):
        super(Pred_label, self).__init__()
        embed_shape = model.embed.weight.shape

        self.FC = nn.Linear(embed_shape[1], embed_shape[1])
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, h):
        h = self.FC(h)
        lm_logits = self.decoder(h)
        lm_logits = self.softmax(lm_logits)
        return lm_logits


class Ensemble_encoder(nn.Module):
    def __init__(self, vocab_num, GAT_num, Trans_num, feat_dim, nhid_gat, nhid_trans, dropout, alpha, nheads, blank, fc = False):
        """Dense version of GAT."""
        super(Ensemble_encoder, self).__init__()
        print("initialize Encoder with GAT num ", GAT_num, " Bert num ", Trans_num)
        self.fc = fc
        if self.fc:
            self.embed = nn.Embedding(vocab_num, feat_dim)
            self.Pred_label = Pred_label(self)
            self.fcs = nn.ModuleList(
                nn.Linear(feat_dim, feat_dim/2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim/2, feat_dim/2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim/2, feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.GAT_Ensemble = GAT_Ensemble(vocab_num, feat_dim, nhid_gat, nhid_trans, dropout, alpha, nheads, GAT_num)
            self.Trans_Ensemble = Transformer_Ensemble(Trans_num, d_word_vec=nhid_trans, d_model=nhid_trans)
            self.Pred_label = Pred_label(self.GAT_Ensemble)

        # self.Pred_label = Pred_label(self)

        print("nhid_trans: ", nhid_trans)
        print("int(nhid_trans/2): ", int(nhid_trans/2))

        self.Pred_connect = Connect_Cls(nhid_trans, int(nhid_trans/2), 2)

        self.blank = blank


    def forward(self, fea, adj):
        slf_attn_mask = get_attn_key_pad_mask(seq_k=fea, seq_q=fea, blank=self.blank)
        non_pad_mask = get_non_pad_mask(fea, blank=self.blank)

        # non_pad_mask = None
        # slf_attn_mask = None

        # x = self.embed(fea)
        if self.fc:
            x = self.fcs(fea)
        else:
            x = self.GAT_Ensemble(fea, adj, non_pad_mask)
            x = self.Trans_Ensemble(x, slf_attn_mask, non_pad_mask)

        pred_label = self.Pred_label(x)
        pred_edge = self.Pred_connect(x, adj)

        return pred_label, pred_edge

