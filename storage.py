import torch
import numpy as np

class Storage:

    def __init__(self, size):
        self.keys = ['states', 'actions',
                     'rewards', 'values',
                     'log_pi', 'entropy',
                       'advantage', 'mean',
                       'returns']
        self.size = size
        self.reset()

    def add(self, data: dict):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)
