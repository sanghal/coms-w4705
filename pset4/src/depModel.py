import os
import sys

import dynet as dy

from decoder import ACTIONS, Decoder
from q1 import NN
from q3 import MyNN


class DepModel:
    def __init__(self, which, model):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        self.actions = ACTIONS
        self.args = {
            'q1': {
                'n_h1': 200,
                'n_h2': 200,
                'batch_size': 1000,
                'opt': dy.AdamTrainer,
                'activation': dy.rectify,
            },
            'q2': {
                'n_h1': 400,
                'n_h2': 400,
                'batch_size': 1000,
                'opt': dy.AdamTrainer,
                'activation': dy.rectify,
            },
            'q3': {
                'n_h1': 400,
                'n_h2': 400,
                'batch_size': 1000,
                'opt': dy.AdamTrainer,
                'activation': dy.rectify,
            },
        }[which]
        self.nn = NN(**self.args) if which != 'q3' else MyNN(**self.args)
        self.nn.load(model)

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.decode()
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        val = self.nn.forward(str_features).npvalue()
        dy.renew_cg()

        return val


if __name__=='__main__':
    which = sys.argv[1]
    model = sys.argv[2]
    input_p = os.path.abspath(sys.argv[3])
    output_p = os.path.abspath(sys.argv[4])
    m = DepModel(which, model)
    Decoder(m.score, m.actions).parse(input_p, output_p)
