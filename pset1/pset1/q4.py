#!/usr/bin/python
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

from utils import refresh, main


def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line:
            fields = line.split(" ")
            ne_tag = fields[-1]
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else:
            yield (None, None)
        l = corpus_file.readline()


def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = []
    for l in corpus_iterator:
            if l == (None, None):
                if current_sentence:
                    yield current_sentence
                    current_sentence = []
                else:
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l)

    if current_sentence:
        yield current_sentence


def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
        w_boundary = (n-1) * [(None, "*")]
        w_boundary.extend(sent)
        w_boundary.append((None, "STOP"))
        ngrams = (tuple(w_boundary[i:i+n]) for i in xrange(len(w_boundary)-n+1))
        for n_gram in ngrams:
            yield n_gram


# Reuse of `count_freq.py`
class BaseHMM(object):
    """
    Stores counts for n-grams and emissions.
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = get_ngrams(sentence_iterator(
                simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            assert len(ngram) == self.n, \
                "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram])
            for i in xrange(2, self.n+1):
                self.ngram_counts[i-1][tagsonly[-i:]] += 1

            if ngram[-1][0] is not None:
                self.ngram_counts[0][tagsonly[-1:]] += 1
                self.emission_counts[ngram[-1]] += 1

            if ngram[-2][0] is None:
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1


class NaiveHMM(BaseHMM):
    """
    A NaiveHMM model, predicts tags only based on unigram max-likelihood.
    """
    def merge_word(self, corpus_file, num_clip=5):
        """
        Merge words if their counts are less than num_clip.
        Write intermediate result file into `4_1.txt`.
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

        # fn = fname(corpus_file.name, 'merged')
        # Hardcode filename instead.
        fn = '4_1.txt'
        with open(fn, 'w') as fout:
            fout.write('\n'.join(lines))

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
        Write predict result into `4_2.txt`.

            :eval_file: input file stream of testing corpus.
        """
        # Hardcode filename instead.
        with open(refresh('4_2.txt'), 'w') as output_file:
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


if __name__ == '__main__':
    main(hmm_model=NaiveHMM)
