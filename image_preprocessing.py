# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:15:13 2019

@author: madha
"""

import cv2
from PIL import Image
import numpy as np
from PIL import Image, ImageFilter


def preprocess(image):

    im = image.convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    # creates white canvas of 45x45 pixels
    newImage = Image.new('L', (45, 45), (255))

    if width > height:  # check which dimension is bigger
        # resize height according to ratio width
        nheight = int(round((45.0/width*height), 0))
        img = im.resize((45, nheight), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)
        # caculate horizontal pozition
        wtop = int(round(((45 - nheight)/2), 0))
        newImage.paste(img, (0, wtop))  # paste resized image
    else:
        # resize width according to ratio height
        nwidth = int(round((45.0/height*width), 0))
        img = im.resize((nwidth, 45), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)
        wleft = int(round(((45 - nwidth)/2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 0))  # paste resized image on

    open_cv_image = np.array(newImage)

    # adding Gaussian BLur
    gaussian_blur = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

    # applying binary thresholding
    threshold_image = cv2.adaptiveThreshold(
        gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)

    # dilating the character
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(threshold_image, kernel, iterations=1)

    # cv2.imshow('dilate', eroded)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return eroded


# image display
# cv2.imshow('image', resize)
# cv2.imshow('thresh', threshold_image)
# cv2.imshow('dilate', dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
