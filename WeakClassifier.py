# coding=utf-8
import numpy as np

from setting import FACE, NON_FACE

class WeakClassifier(object):
    """
    @:param self.direction: when it is 1, the sample will be considered as face if its value less than threshold.
    """
    def __init__(self):
        self.direction   = None
        self.threshold   = None
        self.dimension   = None
        self.weightError = np.inf
    def fit(self, X, W, Y):
        """To minimize the weighted error function
        :param X: A matrix sampleNum * DimensionNum
        :param W: Weight corresponding to each dimension 1*DimensionNum
        :param Y: the label of each sample
        :return: minWeightError
        """
        dimensionNum = X.shape[1]

        FaceWeightSum    = W[np.where(Y ==     FACE)].sum()
        NonFaceWeightSum = W[np.where(Y == NON_FACE)].sum()
        for dim in range(dimensionNum):
            thresholds = set(X[:,dim])
            for threshold in thresholds:
                for direction in [1, -1]:
                    FaceWeightSumBeforeTh    = W[np.intersect1d(np.where(Y==FACE),
                                            np.where(X[:, dim]*direction <  threshold*direction))].sum()
                    NonFaceWeightSumBeforeTh = W[np.intersect1d(np.where(Y==NON_FACE),
                                            np.where(X[:, dim]*direction >= threshold*direction))].sum()

                    tempWeightError = min(FaceWeightSumBeforeTh + (NonFaceWeightSum-NonFaceWeightSumBeforeTh),
                                          NonFaceWeightSumBeforeTh + (FaceWeightSum-FaceWeightSumBeforeTh))
                    if tempWeightError < self.weightError:
                        self.weightError = tempWeightError
                        self.dimension = dim
                        self.direction = direction
                        self.threshold = threshold
        return self.weightError

    def predict(self, X):
        pred = np.ones(X.shape[0]) * NON_FACE
        pred[X[:,self.dimension] * self.direction < self.threshold * self.direction] = FACE

        return pred
