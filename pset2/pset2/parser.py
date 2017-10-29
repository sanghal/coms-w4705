#!/usr/bin/python
"""Main entry of parsers for all questions.

File: parser.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 10/27/2017
"""

from __future__ import print_function

import sys
import json
import copy

from collections import defaultdict

from utils import refresh, count_cfg_freq, timeit


class RareProcessor(object):
    """
    A processor for rare words replacement.
    """
    def __init__(self, train_path, count_path='cfg.counts'):
        self.train_path = train_path
        self.count_path = count_path

        count_cfg_freq(train_path, count_path)
        self._count2dict(train_path, count_path)

    def _count2dict(self, train_path, count_path):
        """
        Read count file into class parameters of dictionary.
        """
        self.nonterm = defaultdict(int)
        self.unary = defaultdict(int)
        self.binary = defaultdict(int)

        with open(train_path) as ftrain:
            self.trees = map(lambda l: json.loads(l.strip()),
                             ftrain.readlines())

        with open(count_path) as fcnt:
            for l in fcnt:
                item = l.strip().split()
                if item[1] == 'NONTERMINAL':
                    key = item[-1]
                    cnt = int(item[0])
                    self.nonterm[key] = cnt
                elif item[1] == 'UNARYRULE':
                    key = tuple(item[-2:])
                    cnt = int(item[0])
                    self.unary[key] = cnt
                elif item[1] == 'BINARYRULE':
                    key = tuple(item[-3:])
                    cnt = int(item[0])
                    self.binary[key] = cnt
                else:
                    raise ValueError('Invalid count file format.')

    def replace_rare_words(self, output_path):
        """
        Replace rare words and output result file.
        """
        def replace(tree, nonrare_words, token='_RARE_'):
            if len(tree) == 2:
                if tree[1] not in nonrare_words:
                    tree[1] = token
            elif len(tree) == 3:
                tree[1] = replace(tree[1], nonrare_words, token)
                tree[2] = replace(tree[2], nonrare_words, token)

            return tree

        word_count = defaultdict(int)
        for key, cnt in self.unary.items():
            word_count[key[1]] += cnt
        self.nonrare_words = [k for k, v in word_count.items() if v >= 5]

        self.new_trees = map(lambda t: replace(t, self.nonrare_words),
                             self.trees)

        with open(refresh(output_path), 'w') as fout:
            strs = map(lambda t: json.dumps(t), self.new_trees)
            fout.write('\n'.join(strs))


class PCFGProcessor(RareProcessor):
    """
    A model for CKF PCFG parsing task.
    """
    def __init__(self, rare_path,
                 train_path='parse_train.dat', count_path='cfg.counts'):
        super(PCFGProcessor, self).__init__(train_path, count_path)

        self.replace_rare_words(rare_path)
        count_cfg_freq(rare_path, count_path)
        self._count2dict(rare_path, count_path)

        self._calc_q()

    def _calc_q(self):
        """
        Calculate q parameters and hashes for better performance.
        """
        self.q_binary = defaultdict(float)
        self.q_unary = defaultdict(float)
        self.X_hash = defaultdict(list)
        self.word_hash = defaultdict(list)

        for rule, cnt in self.binary.items():
            self.q_binary[rule] = cnt * 1. / self.nonterm[rule[0]]
            self.X_hash[rule[0]].append(rule[-2:])

        for rule, cnt in self.unary.items():
            self.q_unary[rule] = cnt * 1. / self.nonterm[rule[0]]
            self.word_hash[rule[1]].append(rule[0])

    def cky(self, s, s_ori): # noqa
        """
        The CKY algorithm using dynamic programming.
        """
        def bp2tree(bp, i, j, X, s_ori):
            rule, s = bp[i, j, X]
            if len(rule) == 3:
                tree = [X, bp2tree(bp, i, s, rule[1], s_ori),
                        bp2tree(bp, s + 1, j, rule[2], s_ori)]
            elif len(rule) == 2:
                assert X == rule[0]
                tree = [X, s_ori[i]]
            else:
                raise ValueError('Invalid rule encountered.')
            return tree

        pi = defaultdict(float)
        bp = defaultdict(tuple)
        n = len(s)

        # Initialization
        for i in range(n):
            nonterm = self.word_hash[s[i]]
            for X in self.nonterm:
                if X in nonterm:
                    pi[i, i, X] = self.q_unary[X, s[i]]
                    bp[i, i, X] = ((X, s[i]), -1)
                else:
                    pi[i, i, X] = 0
                    bp[i, i, X] = ((''), -1)

        # DP
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                for X in self.nonterm:
                    _pi = 0
                    _bp = ()
                    for s in range(i, j):
                        rules = self.X_hash[X]
                        for r in rules:
                            r = (X,) + r
                            cur_pi = self.q_binary[r] * pi[i, s, r[1]] * \
                                pi[s + 1, j, r[2]]
                            if _pi < cur_pi:
                                _pi = cur_pi
                                _bp = (r, s)
                    pi[i, j, X] = _pi
                    bp[i, j, X] = _bp

        if pi[0, n - 1, 'S'] != 0:
            return bp2tree(bp, 0, n - 1, 'S', s_ori)
        else:
            _X = ''
            _max = 0
            for X in self.nonterm:
                if _max < pi[0, n - 1, X]:
                    _max = pi[0, n - 1, X]
                    _X = X
            return bp2tree(bp, 0, n - 1, _X, s_ori)

    def eval_model(self, dev_path, output_path):
        """
        Evaluate model by dev set and output predictions.
        """
        def proc_dev_sentence(s, token='_RARE_'):
            s_ori = copy.copy(s)
            for i, w in enumerate(s):
                if w not in self.nonrare_words:
                    s[i] = token
            return s, s_ori

        with open(dev_path) as fdev:
            sen_chunk = map(lambda s: proc_dev_sentence(s.strip().split()),
                            fdev.readlines())

            strs = []
            for chunk in sen_chunk:
                tree = self.cky(*chunk)
                strs.append(json.dumps(tree))

        with open(output_path, 'w') as fout:
            fout.write('\n'.join(strs))


@timeit
def main():
    if sys.argv[1] == 'q4':
        rp = RareProcessor(sys.argv[2])
        rp.replace_rare_words(sys.argv[3])
    elif sys.argv[1] == 'q5':
        pp = PCFGProcessor(sys.argv[2])
        pp.eval_model(sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'q6':
        pp = PCFGProcessor(sys.argv[2], train_path='parse_train_vert.dat')
        pp.eval_model(sys.argv[3], sys.argv[4])
    else:
        raise ValueError('Invalid args encountered.')


if __name__ == '__main__':
    main()
