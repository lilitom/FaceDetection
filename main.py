# coding=utf-8
from time import time
from features import loadFeatures, saveFeatures
from model import saveModel, loadModel
from numpy import random
from sklearn.metrics import accuracy_score
from adaboost import Adaboost
from detector import Detector
from matplotlib import image
from PIL import Image
def main():
    starttime = time()
    print("loading model...")
    clf = loadModel()
    print("loading image...")
    TEST_IMG = ".\\test\BioID_0000.pgm"
    print("detecting...")
    detector =Detector(clf)
    detector.detectFace(TEST_IMG)


    endtime = time()


    print("cost: " + str(endtime-starttime))
    #






    pass

if __name__ == "__main__":
    main()