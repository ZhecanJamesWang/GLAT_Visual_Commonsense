import numpy as np
import scipy.sparse as sp
import re
import ftfy
import json
import spacy
import torch
from tqdm import tqdm
import pdb
import os


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_model(model, path):

    if path != "":
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc = checkpoint['best_acc']
        print("successfully loading: ", path)
    else:
        epoch = -1
        acc = -1
    return model, epoch, acc

def save_model(model, epoch, filename, model_path, foldername, best_acc):

    diretory = os.path.join(model_path, foldername)

    if not os.path.exists(diretory):
        os.makedirs(diretory)

    ckpt = dict(
        epoch=epoch,
        best_acc=best_acc,
        model=model.state_dict(),
    )

    path = os.path.join(diretory, filename)
    torch.save(ckpt, path)
    print("model saved at: ", path)


class Record(object):
    def __init__(self):
        self.best_test_node_mask_acc = 0
        self.best_test_edge_pos_acc = 0

    def compare_node_mask_acc(self, candidate):
        if candidate > self.best_test_node_mask_acc:
            self.best_test_node_mask_acc = candidate
            return True
        else:
            return False

    def compare_edge_pos_acc(self, candidate):
        if candidate > self.best_test_edge_pos_acc:
            self.best_test_edge_pos_acc = candidate
            return True
        else:
            return False

    def get_best_test_node_mask_acc(self):
        return self.best_test_node_mask_acc

    def get_best_test_edge_pos_acc(self):
        return self.best_test_edge_pos_acc


class Counter(object):
    def __init__(self, classes=2):
        # self.corr_cumul = 0
        # self.num_cumul = 0
        self.classes = classes
        self.correct = [0] * self.classes
        self.num_pred = [0] * self.classes
        self.num_label = [0] * self.classes

    def add(self, pred, labels):

        if pred.size()[-1] != 1 and len(pred.size()) != 1:
            preds = pred.max(1)[1].type_as(labels)
        else:
            preds = pred

        acc = preds == labels

        for i in range(self.classes):
            self.correct[i] += (preds[acc] == i).sum()
            # self.correct[i] += preds[preds == i].eq(labels[labels == i]).double()
            self.num_pred[i] += len(preds[preds == i])
            self.num_label[i] += len(labels[labels == i])
        # correct.append(correct.sum())
        # return correct, len(labels)

    # def add(self, pred, labels):
    #     corr, num = self.cal_accur(pred, labels)
    #     self.corr_cumul += corr
    #     self.num_cumul += num

    def class_acc(self):
        # print("self.correct: ", self.correct)
        # print("np.asarray(self.num_pred): ", np.asarray(self.num_pred))
        return list(self.correct/np.asarray(self.num_pred))

    def overall_acc(self):
        # print("sum(self.correct): ", sum(self.correct))
        # print("sum(self.num_pred)： ", sum(self.num_pred))
        return float(sum(self.correct))/sum(self.num_pred)

    def recall(self):
        return list(self.correct/np.asarray(self.num_label))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load(
            'en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if (word[i] == first and i < len(word) - 1 and
                        word[i+1] == second):
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


