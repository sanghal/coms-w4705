"""Answer to Question 4

File: q4.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 09/23/2017
"""
from __future__ import print_function

from collections import defaultdict
from math import log

from count_freqs import BaseHMM
from utils import refresh, fname, fext, main


class NaiveHMM(BaseHMM):
    """
    A NaiveHMM model, predicts tags only based on unigram max-likelihood.
    """
    def merge_word(self, corpus_file, num_clip=5):
        """
        Merge words if their counts are less than num_clip.
        Write intermediate result file into `*.merged.dat`.
        Record vocabulary and tags in the training set.

            :corpus_file: input file stream of training corpus.
            :num_clip: number of counts to clip.

            :return: result file of merged data.
        """
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

        word_to_merge = [w for w, c in word_counts.items() if c < num_clip]
        for wm in word_to_merge:
            for idx in word_idx[wm]:
                lines[idx] = '_RARE_ {}'.format(lines[idx].split()[-1])

        fn = fname(corpus_file.name, 'merged')
        with open(fn, 'w') as fout:
            fout.write('\n'.join(lines))

        with open('rare.train.log', 'w') as flog:
            flog.write('\n'.join(word_to_merge))

        self.vocab = set(vocab) - set(word_to_merge)
        self.vocab.update(set(['_RARE_']))
        self.tags = set(tags)

        ret = open(fn)
        return ret

    def train(self, corpus_file, num_clip=5):
        """
        Train HMM based on merged training data.

            :corpus_file: input file stream of merged training corpus.
            :num_clip: number of counts to clip.
        """
        merged_corpus_file = self.merge_word(corpus_file, num_clip)
        super(NaiveHMM, self).train(merged_corpus_file)
        self.compute_emission()

    def compute_emission(self):
        """
        Compute emission parameters based on emission counts and n-gram counts.

            :return: emission parameters.
        """
        self.emission = defaultdict(float)
        for (word, tag), count in self.emission_counts.items():
            self.emission[(word, tag)] = count * 1. / \
                self.ngram_counts[0][(tag,)]

        return self.emission

    def eval_model(self, eval_file):
        """
        Evaluate HMM model based on evaluation file.
        Write predict result into `*.baseline.pred`.

            :eval_file: input file stream of testing corpus.
        """
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
            rare_word = []
            for l in eval_file.readlines():
                assert len(l.strip().split(' ')) == 1, 'Illegal develope set.'
                w = l.strip()
                if w:
                    if w not in self.vocab:
                        rare_word.append(w)
                    ww = '_RARE_' if w not in self.vocab else w
                    cache.append('{} {} {}'.format(
                        w, max_emission_tags[ww][1], max_emission_tags[ww][0])
                    )
                else:
                    cache.append('')

            output_file.write('\n'.join(cache) + '\n')

        with open('rare.dev.log', 'w') as frare:
            frare.write('\n'.join(rare_word))


if __name__ == '__main__':
    main(hmm_model=NaiveHMM)
