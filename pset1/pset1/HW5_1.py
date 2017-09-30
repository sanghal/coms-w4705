#!/usr/bin/python
"""Stand-alone Answer to Question 5.1

File: HW5_1.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 09/29/2017
"""
from q5 import TrigramHMM
from utils import timeit


@timeit
def main():
    ftr = open('ner_train.dat', 'r')

    hmm = TrigramHMM()

    hmm.train(ftr)
    hmm.enumerate_trigram()
    hmm.eval_trigram_ml()


if __name__ == '__main__':
    main()
