# coding=utf-8
import os
import multiprocessing
from haar import Haar
from setting import WINDOW_HEIGHT, WINDOW_WIDTH, TRAIN_FACE, TRAIN_NON_FACE, FEATURE_FILE, FACE, NON_FACE
from image import Img
import numpy as np
import threading
import pickle


def loadImage(indexlist, trainFiles, mat, haar):
    while len(indexlist) != 0:
        index = indexlist.pop()
        print(index)
        img = Img(trainFiles[index])
        mat[index, :-1] = haar.calImgFeatureVal(img.integralMat)
        if TRAIN_NON_FACE in trainFiles[index]:
            mat[index,-1] = NON_FACE
        else:
            mat[index,-1] = FACE

def getFeatures():
    """load images and calculate the features
    :return:
    """
    trainNonFaceFiles = [TRAIN_NON_FACE + i for i in os.listdir(TRAIN_NON_FACE)]
    trainFaceFiles = [TRAIN_FACE + i for i in os.listdir(TRAIN_FACE)]

    haar = Haar(WINDOW_WIDTH, WINDOW_HEIGHT)
    featureValMat = np.zeros((len(trainFaceFiles) + len(trainNonFaceFiles), len(haar.features)+1), dtype='float32')

    indexlist = [i for i in range(len(trainFaceFiles)+len(trainNonFaceFiles))]
    for i in range(multiprocessing.cpu_count()):
        t = threading.Thread(target=loadImage, args=(indexlist, trainFaceFiles+trainNonFaceFiles, \
                                                     featureValMat, haar))
        t.start()
    t.join()
    return featureValMat

def saveFeatures():
    """save the features to file
    """
    featureMat = getFeatures()
    featureFile = open(FEATURE_FILE, "wb")
    pickle.dump(featureMat, featureFile, -1)
    featureFile.close()

def loadFeatures():
    """load
    :return:
    """
    featureFile = open(FEATURE_FILE, "rb")
    featureMat = pickle.load(featureFile)
    featureFile.close()
    return featureMat
