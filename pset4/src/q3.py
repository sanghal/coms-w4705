#!/usr/bin/python
"""Answer to Question 3

File: q3.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 12/09/2017
"""
from __future__ import print_function

import dynet as dy
import numpy as np

from q1 import NN, load_dat
from q1 import n_words, n_poses, n_labels, n_actions, word2ind
from q1 import allword2ind, allpos2ind, label2ind


WORD_EMBED_DIM = 100
POS_EMBED_DIM = 32
LABEL_EMBED_DIM = 32
INPUT_DIM = 20 * WORD_EMBED_DIM + 20 * POS_EMBED_DIM + 12 * LABEL_EMBED_DIM

BATCH_SIZE = 1000
N_H1 = 400
N_H2 = 400

EMB_PATH = 'glove.6B/glove.6B.100d.txt'


def load_pretrained_word_emb(emb_path=EMB_PATH):
    with open(emb_path, 'r') as f:
        lines = f.read().strip().split('\n')
        vocab = [line.split()[0] for line in lines]
        emb = [list(map(lambda w: float(w), line.split()[1:]))
               for line in lines]

    return vocab, np.array(emb)


def init_from_word_emb():
    pwords, pemb = load_pretrained_word_emb()
    words = word2ind.keys()
    emb = np.zeros(shape=[n_words, WORD_EMBED_DIM])
    for i, w in enumerate(words):
        if w in pwords:
            for pi, pw in enumerate(pwords):
                if w == pw:
                    emb[i] = pemb[pi, :WORD_EMBED_DIM]
                    break

    return emb


class MyNN(NN):
    def build_model(self):
        emb = init_from_word_emb()
        self.word_emb = self.m.add_lookup_parameters((n_words, WORD_EMBED_DIM))
        self.word_emb.init_from_array(emb)

        self.pos_emb = self.m.add_lookup_parameters((n_poses, POS_EMBED_DIM))
        self.label_emb = self.m.add_lookup_parameters((n_labels, LABEL_EMBED_DIM))

        self.h1_layer = self.m.add_parameters((self.n_h1, INPUT_DIM))
        self.h1_bias = self.m.add_parameters(
            self.n_h1, init=dy.ConstInitializer(0.2))

        self.h2_layer = self.m.add_parameters((self.n_h2, self.n_h1))
        self.h2_bias = self.m.add_parameters(
            self.n_h2, init=dy.ConstInitializer(0.2))

        self.out_layer = self.m.add_parameters((n_actions, self.n_h2))
        self.out_bias = self.m.add_parameters(
            n_actions, init=dy.ConstInitializer(0))

    def forward(self, fea):
        word_ind = [allword2ind(word) for word in fea[:20]]
        pos_ind = [allpos2ind(pos) for pos in fea[20:40]]
        label_ind = [label2ind[label] for label in fea[-12:]]

        w_emb = [self.word_emb[ind] for ind in word_ind]
        p_emb = [self.pos_emb[ind] for ind in pos_ind]
        l_emb = [self.label_emb[ind] for ind in label_ind]

        emb = dy.concatenate(w_emb + p_emb + l_emb)

        h1 = self.activation(self.h1_layer.expr() * emb + self.h1_bias.expr())
        h2 = self.activation(self.h2_layer.expr() * h1 + self.h2_bias.expr())
        # Add dropout before softmax
        # h2 = dy.dropout(h2, 0.5)
        out = self.out_layer.expr() * h2 + self.out_bias.expr()

        return out


if __name__ == '__main__':
    train_dat = load_dat('data/train.data')

    nn = MyNN(N_H1, N_H2, BATCH_SIZE, activation=dy.rectify)

    for ep in range(7):
        print('epoch {}'.format(ep + 1))
        nn.train_epoch(train_dat)
        dy.renew_cg()

    print('Finish training.')
    nn.save('model/q3.model')
