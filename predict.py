# add your imports here
import boundingBox
import predict_function as pf
import sys
import cv2
from PIL import Image, ImageFilter
from sys import argv
from glob import glob
import numpy as np
import os
import pprint
import operator
from keras.models import load_model
from keras.preprocessing import image
from image_preprocessing import preprocess


# definition of classification
sy = ['!',	 '(',	 ')',	 '+',	 '-',	 '0',	 '1',	 '2',	 '3',	 '4',	 '5',	 '6',	 '7',	 '8',	 '9',	 '=',	 'a',	 'alpha',	 'b',	 'beta',	 'c',	 'cos',	 'd',	 'div',	 'e',	 'f',	 'forward_slash',	 'g',	 'gamma',	 'geq',	 'gt',	 'h',
      'i', 'infty',	 'int',	 'j',	 'k',	 'l',	 'leq',	 'lim',	 'log',	 'lt',	 'm',	 'n',	 'neq',	 'o',	 'p',	 'phi',	 'pi',	 'q',	 'r',	 's',	 'sin',	 'sqrt',	 'sum',	 't',	 'tan',	 'theta',	 'mul',	 'u',	 'v',	 'w',	 'x',	 'y',	 'z']

slash_sy = ['tan', 'sqrt', 'mul', 'pi', 'phi', 'theta',  'sin', 'alpha', 'beta', 'gamma',
            'infty', 'leq', 'sum', 'geq', 'neq', 'lim', 'log', 'int', 'frac', 'cos', 'bar', 'div', '^', '_']

variable = [i for i in sy if i not in slash_sy]


class SymPred():
    def __init__(self, prediction, x1, y1, x2, y2):
        """
        <x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
        (x1,y1)
               .--------
               |           |
               |           |
                --------.
                         (x2,y2)
        """
        self.prediction = prediction
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        return self.prediction + '\t' + '\t'.join([
            str(self.x1),
            str(self.y1),
            str(self.x2),
            str(self.y2)])


class ImgPred():
    def __init__(self, image_name, sym_pred_list, latex='LATEX_REPR'):
        """
        sym_pred_list is list of SymPred
        latex is the latex representation of the equation
        """
        self.image_name = image_name
        self.latex = latex
        self.sym_pred_list = sym_pred_list

    def __str__(self):
        res = self.image_name + '\t' + \
            str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
        for sym_pred in self.sym_pred_list:
            res += str(sym_pred) + '\n'
        return res


def predict(image_path):
    # bounding box on given image and sort the symbols by x and y
    test_symbol_list = boundingBox.createSymbol(image_path)
    # print(test_symbol_list)
    test_symbol_list = sorted(test_symbol_list, key=operator.itemgetter(2, 3))
    # print(test_symbol_list)
    pre_symbol_list = []
    model = load_model('resnet without dropout reduced.hdf5')

    # for each symbol image in image list
    for i in range(len(test_symbol_list)):
        test_symbol = test_symbol_list[i]

        # prepare the each symbol image into standard size
        eroded = preprocess(test_symbol[0])

        # predict
        img2 = cv2.merge((eroded, eroded, eroded))

        # cv2.imshow('dila', img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        im = img2.reshape(1, 45, 45, 3)

        result = model.predict_classes(im, batch_size=32)
        print(test_symbol, result)

        # analysis for dot pattern
        if test_symbol[1] != "dot":
            predict_result = sy[result[0]]
        else:
            predict_result = "dot"
        test_symbol = (test_symbol[0], predict_result, test_symbol[2],
                       test_symbol[3], test_symbol[4], test_symbol[5])
        test_symbol_list[i] = test_symbol

    # combine potential part in equation
    updated_symbol_list = pf.update(image_path, test_symbol_list)

    # for each result in result list add it into return list
    for s in updated_symbol_list:
        pre_symbol = SymPred(s[1], s[2], s[3], s[4], s[5])
        pre_symbol_list.append(pre_symbol)

    # predict the latex expression of equation
    equation = pf.toLatex(updated_symbol_list)

    # out put the result
    head, tail = os.path.split(image_path)
    img_prediction = ImgPred(tail, pre_symbol_list, equation)

    return img_prediction


image_folder_path = './data'

image_paths = glob(image_folder_path + '/*png')

results = []

for image_path in image_paths:
    print(image_path)
    impred = predict(image_path)
    results.append(impred)

with open('predictions.txt', 'w') as fout:
    for res in results:
        fout.write(str(res))
