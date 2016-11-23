# coding=utf-8

"""
Haar TYPE

     --- ---
    |   +   |
    |-------|
    |   -   |
     -------
        I
     --- ---
    |   |   |
    | - | + |
    |   |   |
     --- ---
       II

     -- -- --
    |  |  |  |
    |- | +| -|
    |  |  |  |
     -- -- --
       III

     --- ---
    |___-___|
    |___+___|
    |___-___|
       IV

     --- ---
    | - | + |
    |___|___|
    | + | - |
    |___|___|
        V
"""


class Haar(object):
    def __init__(self, img_width, img_height):
        self.IMG_WIDTH  = img_width
        self.IMG_HEIGHT = img_height

        self.WINDOW_WIDTH  = self.IMG_WIDTH
        self.WINDOW_HEIGHT = self.IMG_HEIGHT

        self.HAAR_TYPES = (
            'HAAR_TYPE_I',
            'HAAR_TYPE_II',
            'HAAR_TYPE_III',
            'HAAR_TYPE_IV',
            'HAAR_TYPE_V',
        )

        self.features = []
        self._createFeatures()



    def _createFeatures(self):
        """create all kinds of haar features in this window size
        :return: [(type, x, y, w, h),
                  (type, x, y, w, h),
                  ...]
        """

        WIDTH_LIMIT  = {
            "HAAR_TYPE_I" : self.WINDOW_WIDTH,
        }

        HEIGHT_LIMIT = {
            "HAAR_TYPE_I" : int(self.WINDOW_HEIGHT/2),
        }

        type = "HAAR_TYPE_I"
        # for type in self.HAAR_TYPES:
        for w in range(1, WIDTH_LIMIT[type]+1):
            for h in range(1, HEIGHT_LIMIT[type]+1):

                if w == 1 and h == 1:
                    continue

                if type == "HAAR_TYPE_I":
                    x_limit = self.WINDOW_WIDTH  - w
                    y_limit = self.WINDOW_HEIGHT - 2*h

                    for x in range(0, x_limit+1):
                        for y in range(0, y_limit+1):
                            self.features.append([type, x, y, w, h])

    def calImgFeatureVal(self, IntegralMat):
        """
        :param IntegralMat: the integral value of the image
        :return: a list including values of all features
        """
        pass

