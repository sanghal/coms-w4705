"""Answer to Question 4

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

from count_freqs import BaseHMM
from utils import refresh, fname, fext


def merge_rare_word(corpus_file):
    word_counts = defaultdict(int)
    word_idx = defaultdict(list)

    content = corpus_file.read()
    lines = map(lambda l: l.strip(), content.split('\n'))
    tags = []
    vocab = []

    for idx, line in enumerate(lines):
        if line:
            line = line.split(' ')
            word, tag = ' '.join(line[:-1]), line[-1]
            word_counts[word] += 1
            word_idx[word].append(idx)
            tags.append(tag)
            vocab.append(word)

    word_to_merge = [w for w, c in word_counts.items() if c < 5]
    for wm in word_to_merge:
        for idx in word_idx[wm]:
            lines[idx] = '_RARE_ {}'.format(lines[idx].split()[-1])

    fn = fname(corpus_file.name, 'merged')

    with open(fn, 'w') as fout:
        fout.write('\n'.join(lines))

    vocab = set(vocab) - set(word_to_merge)
    vocab.update(set(['_RARE_']))
    ret = open(fn)
    return ret, vocab, set(tags)


class NaiveHMM(BaseHMM):

    def train(self, corpus_file):
        merged_corpus_file, self.vocab, self.tags = merge_rare_word(corpus_file)
        super(self.__class__, self).train(merged_corpus_file)
        self.compute_emission()

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

    counter = NaiveHMM(3)
    counter.train(ftr)
    counter.eval_baseline(fte)
