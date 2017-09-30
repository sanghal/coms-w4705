#!/usr/bin/python
"""Stand-alone Answer to Question 4.1

File: HW4_1.py
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

    hmm = NaiveHMM()
    hmm.merge_word(ftr)


if __name__ == '__main__':
    main()
