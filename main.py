# coding=utf-8
from time import time
from features import loadFeatures, saveFeatures
from model import saveModel, loadModel, getModel
from numpy import random
from sklearn.metrics import accuracy_score
from adaboost import Adaboost
from detector import Detector
from matplotlib import image
from PIL import Image
def main():
    #TEST
    print("loading model...")
    clf = loadModel()
    starttime = time()

    print("loading image...")
    TEST_IMG = ".\\test\BioID_1197.pgm"
    print("detecting...")

    detector =Detector(clf)
    detector.detectFace(TEST_IMG, _show=False, _save=True)
    endtime = time()
    print("cost: " + str(endtime-starttime))

    #FEATURE
    # starttime = time()
    # featureMat = loadFeatures()
    # print("sss")
    # model = loadModel()
    # data = featureMat[:, :-1]
    # label = featureMat[:, -1].reshape(-1, 1)
    # pred = model.predict(data)
    # print(accuracy_score(label, pred))
    # endtime = time()
    # print("cost:" + str(endtime - starttime))

    #MODEL
    # saveModel()




    pass

if __name__ == "__main__":
    main()