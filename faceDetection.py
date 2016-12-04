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
