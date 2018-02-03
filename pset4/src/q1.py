#!/usr/bin/python
"""Answer to Question 1

File: q1.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 12/09/2017
"""
from __future__ import print_function

import random

import dynet as dy

from collections import defaultdict


def load_vocab(vocab_path):
    key2ind = defaultdict(int)

    with open(vocab_path, 'r') as f:
        for line in f.read().strip().split('\n'):
            key, ind = line.split()
            ind = int(ind)
            key2ind[key] = ind

    keys = set(key2ind.keys())
    ind2key = {ind: key for key, ind in key2ind.items()}

    return keys, key2ind, ind2key


words, word2ind, ind2word = load_vocab('data/vocabs.word')
poses, pos2ind, ind2pos = load_vocab('data/vocabs.pos')
labels, label2ind, ind2label = load_vocab('data/vocabs.labels')
actions, action2ind, ind2action = load_vocab('data/vocabs.actions')

n_words = len(words)
n_poses = len(poses)
n_labels = len(labels)
n_actions = len(actions)


def allword2ind(word):
    return (word2ind[word] if word in words else word2ind['<unk>'])


def allpos2ind(pos):
    return (pos2ind[pos] if pos in poses else pos2ind['<null>'])


def load_dat(dat_path):
    with open(dat_path, 'r') as f:
        lines = f.read().strip().split('\n')

    return [line.split() for line in lines]


WORD_EMBED_DIM = 64
POS_EMBED_DIM = 32
LABEL_EMBED_DIM = 32
INPUT_DIM = 20 * WORD_EMBED_DIM + 20 * POS_EMBED_DIM + 12 * LABEL_EMBED_DIM

BATCH_SIZE = 1000
N_H1 = 200
N_H2 = 200


class NN(object):
    def __init__(self, n_h1, n_h2, batch_size,
                 opt=dy.AdamTrainer, activation=dy.rectify):
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.batch_size = batch_size

        self.losses = []

        self.m = dy.Model()
        self.opt = opt(self.m)

        self.activation = activation

        self.build_model()

    def build_model(self):
        self.word_emb = self.m.add_lookup_parameters((n_words, WORD_EMBED_DIM))
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
        out = self.out_layer.expr() * h2 + self.out_bias.expr()

        return out

    def train_epoch(self, train_dat):
        batch_losses = []
        random.shuffle(train_dat)

        for sample in train_dat:
            fea, action = sample[:-1], sample[-1]
            # print(fea, '\n', action)
            action_ind = action2ind[action]
            # print(action, action_ind)

            out = self.forward(fea)
            cur_loss = dy.pickneglogsoftmax(out, action_ind)
            batch_losses.append(cur_loss)

            if len(batch_losses) >= self.batch_size:
                total_loss = dy.esum(batch_losses) / len(batch_losses)
                total_loss.forward()
                total_loss_val = total_loss.value()

                self.losses.append(total_loss_val)

                if len(self.losses) % 10 == 0:
                    print('Loss: {}'.format(total_loss_val))

                total_loss.backward()
                self.opt.update()

                batch_losses = []
                dy.renew_cg()

        dy.renew_cg()

    def load(self, model_path):
        self.m.populate(model_path)

    def save(self, model_path):
        self.m.save(model_path)


if __name__ == '__main__':
    train_dat = load_dat('data/train.data')

    nn = NN(N_H1, N_H2, BATCH_SIZE)

    for ep in range(7):
        print('epoch {}'.format(ep + 1))
        nn.train_epoch(train_dat)
        dy.renew_cg()

    print('Finish training.')
    nn.save('model/q1.model')
