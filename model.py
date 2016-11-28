# coding=utf-8
from features import loadFeatures
from adaboost import Adaboost
from setting import MODEL_CACHE_FILE
from numpy import random
import pickle

def getModel():
    featureMat = loadFeatures()
    print("features loading over...")
    random.shuffle(featureMat)

    train_data  = featureMat[:1500, :-1]
    train_label = featureMat[:1500,  -1].reshape(-1, 1)

    clf = Adaboost(n_estimators=130, debug=True)
    clf.fit(train_data, train_label)

    return clf

def saveModel():
    """save trained model to the file
    :return:
    """
    model = getModel()
    modelFile = open(MODEL_CACHE_FILE, "wb")
    pickle.dump(model, modelFile, -1)
    modelFile.close()

def loadModel():
    modelFile = open(MODEL_CACHE_FILE, "rb")
    model = pickle.load(modelFile)
    modelFile.close()

    return model