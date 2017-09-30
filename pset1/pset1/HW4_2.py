#!/usr/bin/python
"""Stand-alone Answer to Question 4.2

File: HW4_2.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 09/29/2017
"""
from q4 import NaiveHMM
from utils import timeit


@timeit
def main():
    ftr = open('ner_train.dat', 'r')
    fte = open('ner_dev.dat', 'r')

    hmm = NaiveHMM()

    hmm.train(ftr)
    hmm.eval_model(fte)


if __name__ == '__main__':
    main()
