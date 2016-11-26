# coding=utf-8
import numpy as np


class Adaboost(object):
    def __init__(self, n_estimators = 100):
        self.n_estimators = 100
        self.weakClassifiers = [None for i in range(self.n_estimators)]
        self.alpha = np.zeros(n_estimators)

