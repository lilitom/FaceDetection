# coding=utf-8

from haar import Haar
from setting import WINDOW_HEIGHT, WINDOW_WIDTH
from image import Img

def main():
    haar = Haar(WINDOW_WIDTH, WINDOW_HEIGHT)
    img = Img("/home/ivan/Documents/FaceDetection/train/face/face00001.bmp")
    img.calHaarFeatures()

    pass

if __name__ == "__main__":
    main()