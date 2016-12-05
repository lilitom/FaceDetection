# coding=utf-8
from time import time
from features import loadFeatures, calAndSaveFeatures
from setting import TEST
from model import calAndSaveModel, loadModel, getModel
from numpy import random
from sklearn.metrics import accuracy_score
from adaboost import Adaboost
from detector import Detector
from matplotlib import image
from PIL import Image
import os

def main():

    # Feature
    # calAndSaveFeatures()

    # MODEL
    # calAndSaveModel()

    # TEST
    starttime = time()

    print("loading model...")
    clf = loadModel()
    detector =Detector(clf)

    print("detecting...")
    index = 0
    for i in os.listdir(TEST)[1328:]:
        print("detecting " + str(index) + "th...")
        print(TEST+i)
        detector.detectFace(TEST + i, _show=False, _save=True, _saveInfo=True)
        index = index + 1

    endtime = time()
    print("cost: " + str(endtime-starttime))



if __name__ == "__main__":
    main()