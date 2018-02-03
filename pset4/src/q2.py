#!/usr/bin/python
"""Answer to Question 2

File: q2.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 12/09/2017
"""
from __future__ import print_function

import dynet as dy

from q1 import NN, load_dat


BATCH_SIZE = 1000
N_H1 = 400
N_H2 = 400


if __name__ == '__main__':
    train_dat = load_dat('data/train.data')

    nn = NN(N_H1, N_H2, BATCH_SIZE)

    for ep in range(7):
        print('epoch {}'.format(ep + 1))
        nn.train_epoch(train_dat)
        dy.renew_cg()

    print('Finish training.')
    nn.save('model/q2.model')
