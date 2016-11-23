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
        s = self.mat.copy().astype("float32")
        for i in range(1, s.shape[0]):
            s[i,:] = s[i,:] + s[i-1,:]

        self.integralMat = np.zeros(self.mat.shape).astype("float32")
        for i in range(0, self.integralMat.shape[1]):
            if i == 0:
                self.integralMat[:,i] = s[:,i]
            else:
                self.integralMat[:,i] = s[:,i] + self.integralMat[:,i-1]

    def calHaarFeatures(self, haar):
        self.features = haar.calImgFeatureVal(self.integralMat)

    def show(self):
        Image.fromarray(self.mat).show()