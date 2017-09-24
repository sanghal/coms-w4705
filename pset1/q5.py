"""Answer to Question 5

File: q4.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 09/23/2017
"""
from __future__ import print_function

import sys

from collections import defaultdict
from math import log

from q4 import NaiveHMM
from utils import refresh, fname, fext


class TrigramHMM(NaiveHMM):

    def compute_emission(self):
        self.emission = defaultdict(float)
        for (word, tag), count in self.emission_counts.items():
            self.emission[(word, tag)] = count * 1. / \
                self.ngram_counts[0][(tag,)]

        return self.emission

    def eval_baseline(self, eval_file):
        with open(refresh(fext(
            eval_file.name, 'baseline.pred')), 'w'
        ) as output_file:
            max_emission_tags = defaultdict(tuple)
            for w in self.vocab:
                max_emi = float('-inf')
                max_t = ''
                for t in self.tags:
                    # TODO: log 2 or 10?
                    if self.emission[w, t] != 0 and \
                            max_emi < log(self.emission[w, t], 2):
                        max_emi = log(self.emission[w, t], 2)
                        max_t = t
                max_emission_tags[w] = (max_emi, max_t)

            cache = []
            for l in eval_file.readlines():
                assert len(l.strip().split(' ')) == 1, 'Illegal develope set.'
                w = l.strip()
                if w:
                    ww = '_RARE_' if w not in self.vocab else w
                    cache.append('{} {} {:.5f}'.format(
                        w, max_emission_tags[ww][1], max_emission_tags[ww][0])
                    )
                else:
                    cache.append('')

            output_file.write('\n'.join(cache) + '\n')


if __name__ == '__main__':
    try:
        ftr = open(sys.argv[1], 'r')
        fte = open(sys.argv[2], 'r')
    except IOError:
        sys.stderr.write('ERROR: Cannot read inputfile %s.\n'.format(sys.argv))
        sys.exit(1)

    counter = TrigramHMM(3)
    counter.train(ftr)
    counter.eval_baseline(fte)
