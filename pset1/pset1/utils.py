#!/usr/bin/python
"""Utilities to keep code clean.

File: utils.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 09/23/2017
"""

from __future__ import print_function

import sys
import os
import time


def refresh(path):
    """Refresh file if exists by removing it."""
    if os.path.exists(path):
        os.remove(path)
    return path


def fname(old_name, sep):
    """Filename factory to insert `sep` before final extension."""
    old = old_name.split('.')
    old.append(sep)
    old[-1], old[-2] = old[-2], old[-1]
    return '.'.join(old)


def fext(old_name, ext):
    """Filename factory to change former extension into `ext`."""
    old = old_name.split('.')
    old[-1] = ext
    return '.'.join(old)


def timeit(func):
    """Timer decorator for computation performance evaluation."""
    def inner(*args, **kwargs):
        time_start = time.time()
        ret = func(*args, **kwargs)
        time_end = time.time()
        print('**** With total running time of {:.2f}s'.format(
            time_end - time_start
        ))
        return ret
    return inner


@timeit
def main(hmm_model, *args, **kwargs):
    """
    Main runner for questions.

        [training corpus] => train => [testing corpus] => eval =>
        [evaluation results with logits] => (outer eval => [final performance])
    """
    try:
        ftr = open(sys.argv[1], 'r')
        fte = open(sys.argv[2], 'r')
    except IOError:
        sys.stderr.write('ERROR: Cannot read inputfile %s.\n'.format(sys.argv))
        sys.exit(1)

    counter = hmm_model(3, *args, **kwargs)
    counter.train(ftr)

    counter.eval_model(fte)
