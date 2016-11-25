# coding=utf-8

from haar import Haar
from setting import WINDOW_HEIGHT, WINDOW_WIDTH
from image import Img
from time import time
from matplotlib import image
from PIL import Image
def main():
    starttime = time()
    haar = Haar(WINDOW_WIDTH, WINDOW_HEIGHT)

    img = Img("./train/face/face00001.bmp")
    featureVal = haar.calImgFeatureVal(img.integralMat)
    endtime = time()
    print(haar.features[-1])
    print(featureVal[-1])
    print(img.mat)
    print(endtime-starttime)



    pass

if __name__ == "__main__":
    main()