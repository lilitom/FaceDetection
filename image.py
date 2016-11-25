# coding=utf-8
from matplotlib import image
from PIL import Image
import numpy as np
from haar import Haar

class Img(object):
    def __init__(self, fileName=None):

        self.mat         = None
        self.integralMat = None
        self.label       = None

        if fileName is not None:
            self.mat = image.imread(fileName)
            self._calIntegralMat()

        self.WIDTH  = self.mat.shape[1]
        self.HEIGHT = self.mat.shape[0]

    def _calIntegralMat(self):
        self.integralMat = self.mat.astype('float64').cumsum(axis=1).cumsum(axis=0)


    def show(self):
        Image.fromarray(self.mat).show()