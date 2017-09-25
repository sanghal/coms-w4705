"""Answer to Question 5

File: q5.py
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
from utils import refresh, fext, main


class TrigramHMM(NaiveHMM):
    """
    A TrigramHMM model, predicts tags using Hidden Markov Model algorithm with
    first order trigram MLE.
    """
    def train(self, corpus_file, num_clip=5):
        """
        Train HMM based on merged training data.

            :corpus_file: input file of merged training corpus.
            :num_clip: number of counts to clip.
        """
        super(TrigramHMM, self).train(corpus_file, num_clip)
        self.compute_ngram_ml()

    def compute_ngram_ml(self):
        """
        Compute max-likelihood `q` parameters based on bigram and trigram
        counts.
        """
        self.ngram_ml = defaultdict(float)
        for ngram, count in self.ngram_counts[self.n - 1].items():
            w_rest = ngram[:-1]
            self.ngram_ml[ngram] = count * 1. / \
                self.ngram_counts[self.n - 2][w_rest]

    def enumerate_trigram(self, fout=open('ner_train.trigram', 'w')):
        """
        Enumerate possible trigrams in the training corpus.
        Write result into out stream `fout`.

            :fout: output file stream.
        """
        assert self.n > 2, 'Cannot compute trigram with n < 3.'
        lines = []
        for trigram in self.ngram_counts[2]:
            line = ' '.join(trigram)
            lines.append(line)

        fout.write('\n'.join(lines))
        fout.close()

    def eval_trigram_ml(self,
                        trigram_file=open('ner_train.trigram'),
                        fout=sys.stdout):
        """
        Evaluate trigram max-likelihood parameters given an input trigram file
        stream.
        Write result into out stream `fout`.

            :trigram_file: input trigram file steam.
            :fout: output file stream.
        """
        liter = trigram_file.readline()
        while(liter):
            w, u, v = liter.strip().split(' ')
            fout.write('{} {} {} {}'.format(
                w, u, v, log(self.ngram_ml[(w, u, v)])))
            liter = trigram_file.readline()
            if liter:
                fout.write('\n')

        trigram_file.close()

    def viterbi(self, sentence):
        """
        Implementation of Viterbi Algorithm with Back-pointers of log-prob.
        The MLE is now based on first-order estimation.

            :sentence: a sentence for tagging.

            :return: a list of tags & a list of log-prob assign to each words
                in the `sentence`
        """
        def tagset(i):
            return self.tags if i > -1 else ['*']

        n = len(sentence)
        pi = defaultdict(float)
        pi[(-1, '*', '*')] = 1

        bp = defaultdict(float)

        for i in range(n):
            for u in tagset(i - 1):
                for v in tagset(i):
                    max_pi = float('-inf')
                    max_bp = ''
                    for w in tagset(i - 2):
                        if self.ngram_ml[(w, u, v)] * \
                                self.emission[sentence[i], v] != 0:
                            cur_pi = pi[(i - 1, w, u)] + \
                                log(self.ngram_ml[(w, u, v)], 2) + \
                                log(self.emission[sentence[i], v], 2)
                            if max_pi < cur_pi:
                                max_pi = cur_pi
                                max_bp = w

                    pi[(i, u, v)] = max_pi
                    bp[(i, u, v)] = max_bp

        ret_tags = ['*'] * 2 + [''] * n
        max_pi = float('-inf')
        max_u = ''
        max_v = ''
        for u in tagset(i - 1):
            for v in tagset(i):
                if self.ngram_ml[(u, v, 'STOP')] != 0:
                    cur_pi = pi[(n - 1, u, v)] + \
                        log(self.ngram_ml[(u, v, 'STOP')], 2)
                    if max_pi < cur_pi:
                        max_pi = cur_pi
                        max_u = u
                        max_v = v
        ret_tags[-1] = max_v
        ret_tags[-2] = max_u

        for i in range(n - 1, -1, -1):
            ret_tags[i] = bp[(i, ret_tags[i + 1], ret_tags[i + 2])]

        return ret_tags[2:], \
            [pi[(i, ret_tags[i + 1], ret_tags[i + 2])] for i in range(n)]

    def eval_model(self, eval_file, ext='trivit'):
        """
        Evaluate HMM model based on evaluation file.
        Write predict result into `*.trivit.pred`.

            :eval_file: input file stream of testing corpus.
            :ext: output file extension or indicator.
        """
        ext += '.pred'

        with open(refresh(fext(eval_file.name, ext)), 'w') as output_file:
            cache = []
            sentence = []
            raw_sentence = []

            for l in eval_file.readlines():
                assert len(l.strip().split(' ')) == 1, 'Illegal develope set.'
                w = l.strip()
                if w:
                    ww = '_RARE_' if w not in self.vocab else w
                    sentence.append(ww)
                    raw_sentence.append(w)
                else:
                    tags, logits = self.viterbi(sentence)
                    for i in range(len(sentence)):
                        cache.append(
                            '{} {} {}'.format(
                                raw_sentence[i], tags[i], logits[i]))
                    sentence = []
                    raw_sentence = []
                    cache.append('')

            output_file.write('\n'.join(cache) + '\n')


if __name__ == '__main__':
    main(hmm_model=TrigramHMM)
