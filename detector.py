# coding=utf-8
from PIL import Image
from matplotlib import image
from setting import WINDOW_WIDTH, WINDOW_HEIGHT, FACE, NON_FACE
from image import Img
import  numpy as np
from adaboost import Adaboost
from weakClassifier import WeakClassifier
from haar import Haar

from setting import WINDOW_HEIGHT, WINDOW_WIDTH

class Detector(object):
    def __init__(self, model):

        self.DETECT_START = 1.
        self.DETECT_END   = 9.
        self.DETECT_STEP  = 1
        self.DETECT_STEP_FACTOR = 4

        self.haar = Haar(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.model = model
        self.selectedFeatures = [None for i in range(model.n_estimators)]
        self._selectFeatures()


    def _selectFeatures(self):
        for i in range(self.model.n_estimators):
            #[[x, y, w, h, dimension],...]
            self.selectedFeatures[i] = self.haar.features[self.model.weakClassifiers[i].dimension] \
                                       + [self.model.weakClassifiers[i].dimension]



    def detectFace(self, fileName):
        img = Img(fileName, calIntegral=False)

        scaledWindows = []
        #[[window_x, window_y, window_w, window_h, window_scale],...]
        for scale in np.arange(self.DETECT_START, self.DETECT_END, self.DETECT_STEP):
            self._detectInScale(scale, img, scaledWindows)

        scaledWindows = np.array(scaledWindows)

        # detect whether the scaledWindows are face
        faceWindows = self._detectScaledWindow(scaledWindows, img)
        self.show(img.mat, faceWindows)

    def show(self, imageMat, faceWindows):
        for i in range(len(faceWindows)):
            window_x, window_y, window_w, window_h, scale = faceWindows[i]
            self._drawLing(imageMat, int(window_x), int(window_y), int(window_w), int(window_h))
        Image.fromarray(imageMat).show()

    def _drawLing(self, imageMat, x, y, w, h):
        """draw the boundary of the face in the image
        """
        imageMat[y,     x:x+w] = 0
        imageMat[y+h,   x:x+w] = 0
        imageMat[y:y+h, x    ] = 0
        imageMat[y:y+h, x+w  ] = 0


    def _detectInScale(self, scale, img, scaledWindows):
        SCALED_WINDOW_WIDTH  = int(WINDOW_WIDTH  * scale)
        SCALED_WINDOW_HEIGHT = int(WINDOW_HEIGHT * scale)

        scaled_window_x_limit = img.WIDTH  - SCALED_WINDOW_WIDTH
        scaled_window_y_limit = img.HEIGHT - SCALED_WINDOW_HEIGHT

        step = int(SCALED_WINDOW_WIDTH/self.DETECT_STEP_FACTOR)

        for x in range(0, scaled_window_x_limit, step):
            for y in range(0, scaled_window_y_limit, step):
                scaledWindows.append((x, y, SCALED_WINDOW_WIDTH, SCALED_WINDOW_HEIGHT, scale))

    def _detectScaledWindow(self, scaledWindows, img):
        scaledWindowsMat = np.zeros((scaledWindows.shape[0], len(self.haar.features)), dtype='float32')

        for window in range(scaledWindows.shape[0]):
            window_x, window_y, window_w, window_h, scale = scaledWindows[window]

            window_x, window_y, window_w, window_h = int(window_x), int(window_y), int(window_w), int(window_h)

            subWindowImgIntegral = Img(mat=img.mat[window_y : window_y+window_h, \
                                           window_x : window_x+window_w]).integralMat
            for f in range(len(self.selectedFeatures)):
                type, x, y, w, h, dimension = self.selectedFeatures[f]
                x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

                if type == "HAAR_TYPE_I":
                    pos = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    neg = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg)/(w * h * 2)
                elif type == "HAAR_TYPE_II":
                    neg = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg)/(w * h * 2)
                elif type == "HAAR_TYPE_III":
                    neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos  = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)
                    neg2 = self.haar.getPixelValInIntegralMat(x + 2 * w, y, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg1 - neg2)/(w * h * 3)

                elif type == "HAAR_TYPE_IV":
                    neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos  = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                    neg2 = self.haar.getPixelValInIntegralMat(x, y + 2 * h, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg1 - neg2)/(w * h * 3)

                elif type == "HAAR_TYPE_V":
                    neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos1 = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)
                    pos2 = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                    neg2 = self.haar.getPixelValInIntegralMat(x + w, y + h, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos1 + pos2 - neg1 - neg2) / (w * h * 4)

        pred = self.model.predict(scaledWindowsMat)
        index = np.where(pred == FACE)
        print(scaledWindows.shape)
        print(scaledWindows[np.where(pred == FACE)[0]].shape)

        return scaledWindows[np.where(pred == FACE)[0]]



