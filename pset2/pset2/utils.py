#!/usr/bin/python
"""Utilities to keep code clean.

File: utils.py
Author: Hang Gao
Uni: hg2469
Email: hang.gao@columbia.edu
Created Date: 10/27/2017
"""

from __future__ import print_function

import os
import time


def refresh(path):
    """Refresh file if exists by removing it."""
    if os.path.exists(path):
        os.remove(path)
    return path


def count_cfg_freq(train_path, count_path='cfg.counts'):
    exc = 'python2 count_cfg_freq.py {} > {}'.format(train_path, count_path)
    os.system(exc)
    return


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
