# -*- coding: utf-8 -*-
"""
Created on Tue May  7 00:24:33 2019

@author: madha
"""

from keras.models import load_model
import numpy as np
import cv2
import os

def preprocess(image):
	#loading image
	#img = cv2.imread('./predict/sqrt.jpg', 0)

	#image resizing
	height, width = image.shape[:2]

	if height != 45:
		height=45

	if width != 45:
		width=45

	resize = cv2.resize(image,(width, height), interpolation = cv2.INTER_LINEAR)

	cv2.imshow('resize',resize)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#adding Gaussian BLur
	blur5 = cv2.GaussianBlur(resize,(5,5), 0)

	#applying inverse binary thresholding
	h3 = cv2.adaptiveThreshold(blur5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,4)

	#dilating the character
	kernel = np.ones((2,2),np.uint8)
	eroded = cv2.erode(h3, kernel, iterations = 1)

	return eroded

def pred(model, img):

	#process the image
	im = preprocess(img)

	img2 = cv2.merge((im,im,im))

	cv2.imshow('pred',im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	im = img2.reshape(1, 45, 45, 3)

	result = model.predict_classes(im, batch_size=32)
	return result


#load model
model = load_model('resnet without dropout reduced.hdf5')


for d in sorted(os.listdir('C:\\Users\\madha\\Desktop\\Handwritten-Equation-Recognition-Tensorflow-master - Copy\\data\\annotated_test_Equal_boxes')):
    img = cv2.imread('C:\\Users\\madha\\Desktop\\Handwritten-Equation-Recognition-Tensorflow-master - Copy\\data\\annotated_test_Equal_boxes\\' + d, 0)
    
    cv2.imshow('read', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(d, pred(model, img))