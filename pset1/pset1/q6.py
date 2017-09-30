#!/usr/bin/python
"""Answer to Question 6

File: q6.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 09/24/2017
"""

from __future__ import print_function

import sys

from collections import defaultdict

from q5 import TrigramHMM
from utils import main, refresh


class ReparsedHMM(TrigramHMM):
    """
    A ReparsedHMM model, inherit from TrigramHMM but reparse word divisions for
    rare data.
    """
    def __init__(self, n=3, num_clip=5):
        super(ReparsedHMM, self).__init__(n=n)
        self.num_clip = num_clip

    def merge_word(self, corpus_file, num_clip):
        """
        Merge words based on a new set of rules:
            `_NUM_`: general numbers, e.g. 4, 1,800, .63 ...
            `_CAP_PARTIAL_`: title-alike words with only the
                first letter to be captital, e.g. Shi-Ting, New York, Allen ...
            `_CAP_TOTAL_`: all-capital words, e.g. COLUMBIA, X-Y-Z ...
            `_TIME_`: time-alike words, note it should be divided
                from `_NUN_`, e.g. 12/42-12, 18:00 ...

        Write intermediate result file into `6_0.txt`.
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

        # Start merging for the RARE words
        ruleset = {
            '_NUM_': [],
            '_CAP_PARTIAL_': [],
            '_CAP_TOTAL_': [],
            '_TIME_': [],
            '_OTHER_': [],
        }

        def merge_by_rule(s):
            def isnumber(s):
                try:
                    s.replace(',', '')
                    float(s)
                    return True
                except ValueError:
                    return False

            def istime(s):
                try:
                    tmp = ''.join(e for e in s if e.isalnum())
                    float(tmp)
                    return True
                except ValueError:
                    return False

            if s.isupper():
                ruleset['_CAP_TOTAL_'].append(s)
            elif s.istitle():
                ruleset['_CAP_PARTIAL_'].append(s)
            elif isnumber(s):
                ruleset['_NUM_'].append(s)
            else:
                if istime(s):
                    ruleset['_TIME_'].append(s)
                else:
                    ruleset['_OTHER_'].append(s)

        map(lambda x: merge_by_rule(x), word_to_merge)

        for key, words in ruleset.items():
            for word in words:
                for idx in word_idx[word]:
                    lines[idx] = '{} {}'.format(key, lines[idx].split()[-1])

        fn = '6_0.txt'
        with open(fn, 'w') as fout:
            fout.write('\n'.join(lines))

        self.vocab = set(vocab) - set(word_to_merge)
        self.vocab.update(ruleset.keys())
        self.tags = set(tags)

        ret = open(fn)
        return ret

    def train(self, corpus_file):
        """
        Train HMM based on merged training data.

            :corpus_file: input file of merged training corpus.
        """
        super(ReparsedHMM, self).train(corpus_file, self.num_clip)

    def eval_model(self, eval_file):
        """
        Evaluate HMM model based on evaluation file.
        Write predict result into `6.txt`.

            :eval_file: input file stream of testing corpus.
        """
        def reparse(s):
            def isnumber(s):
                try:
                    s.replace(',', '')
                    float(s)
                    return True
                except ValueError:
                    return False

            def istime(s):
                try:
                    tmp = ''.join(e for e in s if e.isalnum())
                    float(tmp)
                    return True
                except ValueError:
                    return False

            if s.isupper():
                return '_CAP_TOTAL_'
            elif s.istitle():
                return '_CAP_PARTIAL_'
            elif isnumber(s):
                return '_NUM_'
            else:
                if istime(s):
                    return '_TIME_'
                else:
                    return '_OTHER_'

        with open(refresh('6.txt'), 'w') as output_file:
            cache = []
            sentence = []
            raw_sentence = []

            for l in eval_file.readlines():
                assert len(l.strip().split(' ')) == 1, 'Illegal develope set.'
                w = l.strip()
                if w:
                    ww = reparse(w) if w not in self.vocab else w
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
    main(hmm_model=ReparsedHMM, num_clip=int(sys.argv[3]))
