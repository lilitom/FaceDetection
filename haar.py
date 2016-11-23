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
        self.HAAR_TYPES = (
            'HAAR_TYPE_I',
            'HAAR_TYPE_II',

        )
