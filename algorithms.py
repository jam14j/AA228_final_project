import numpy as np

def Q_learning(Q, sample, gamma=.5, lr=.5):
    # sample = {'s':val, 'a': val, 'r': val, 'sp': val}
    Q[sample['s'],sample['a']] = ((1-lr)*Q[sample['s'],sample['a']] +
                                   lr*(-1*sample['r'] + gamma*[sample['sp'],:].max()))
