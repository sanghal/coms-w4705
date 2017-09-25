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
from utils import fname, main, fext, refresh


class ReparsedHMM(TrigramHMM):
    """
    A ReparsedHMM model, inherit from TrigramHMM but reparse word divisions for
    unseen data.
    """
    def __init__(self, n=3, num_clip=5):
        super(ReparsedHMM, self).__init__(n=n)
        self.num_clip = num_clip

    def merge_word(self, corpus_file, num_clip):
        """
        Merge words based on a new set of rules:
            '_ABBR_': abbreviation ...
            ...
        Write intermediate result file into `*.merged.dat`.
        Record vocabulary and tags in the training set.

            :corpus_file: input file stream of training corpus.
            :num_clip: number of counts to clip.

            :return: result file of merged data.
        """
        ruleset = {
            '_ABBR_': [],
            '_SPEC_': [],
            '_NUM_': [],
            '_CAP_PARTIAL_': [],
            '_CAP_TOTAL_': [],
            '_RARE_': [],
        }

        def merge_by_rule(s):
            if len(s) >= 2 and all(w.isupper() for w in s[:-1]) \
                    and s[-1] == '.':
                ruleset['_ABBR_'].append(s)
            elif all(not w.isalnum() for w in s):
                ruleset['_SPEC_'].append(s)
            else:
                s = ''.join(e for e in s if e.isalnum())
                if s.isdigit():
                    ruleset['_NUM_'].append(s)
                elif s.istitle():
                    ruleset['_CAP_PARTIAL_'].append(s)
                elif all(w.isupper() for w in s):
                    ruleset['_CAP_TOTAL_'].append(s)
                else:
                    return 0
            return 1

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
                word_idx[word].append(idx)
                if not merge_by_rule(word):
                    word_counts[word] += 1
                tags.append(tag)
                vocab.append(word)

        ruleset['_RARE_'] = [w for w, c in word_counts.items() if c < num_clip]

        for key, words in ruleset.items():
            for word in words:
                for idx in word_idx[word]:
                    lines[idx] = '{} {}'.format(key, lines[idx].split()[-1])

        fn = fname(corpus_file.name, 'merged')
        with open(fn, 'w') as fout:
            fout.write('\n'.join(lines))

        # print(ruleset)
        word_to_merge = reduce(lambda x, y: x + y, ruleset.values())
        # print(word_to_merge)
        with open('rare.train.log', 'w') as flog:
            flog.write('\n'.join(word_to_merge))

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

    def eval_model(self, eval_file, ext='trivit.reparse'):
        """
        Evaluate HMM model based on evaluation file.
        Write predict result into `*.reparse.pred`.

            :eval_file: input file stream of testing corpus.
            :ext: output file extension or indicator.
        """
        def reparse(s):
            if len(s) >= 2 and all(w.isupper() for w in s[:-1]) \
                    and s[-1] == '.':
                return '_ABBR_'
            elif all(not w.isalnum() for w in s):
                return '_SPEC_'
            else:
                s = ''.join(e for e in s if e.isalnum())
                if s.isdigit():
                    return '_NUM_'
                elif s.istitle():
                    return '_CAP_PARTIAL_'
                elif all(w.isupper() for w in s):
                    return '_CAP_TOTAL_'
                else:
                    return '_RARE_'

        ext += '.pred'

        with open(refresh(fext(eval_file.name, ext)), 'w') as output_file:
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


class ClippedHMM(TrigramHMM):
    """
    A ClippedHMM model, deliberate to find optimal counts for clipping.
    """
    def __init__(self, n=3, num_clip=5):
        super(ClippedHMM, self).__init__(n=n)
        self.num_clip = num_clip

    def train(self, corpus_file):
        super(ClippedHMM, self).train(corpus_file, self.num_clip)

    def eval_model(self, eval_file, ext='trivit.clip{}'):
        ext = ext.format(self.num_clip)
        super(ClippedHMM, self).eval_model(eval_file, ext)


if __name__ == '__main__':
    main(hmm_model=ReparsedHMM, num_clip=int(sys.argv[3]))
    main(hmm_model=ClippedHMM, num_clip=int(sys.argv[3]))
