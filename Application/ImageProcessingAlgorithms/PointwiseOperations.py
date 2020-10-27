"""
Module docstring?
"""
import numpy
import math

from Application.Utils.AlgorithmDecorators import RegisterAlgorithm


@RegisterAlgorithm("Invert", "PointwiseOp")
def invert(image):
    """Inverts every pixel of the image.

    :param image:
    :return:
    """
    return {
        'processedImage': numpy.invert(image)
    }


def reverseTransform(c, f):

    g = math.exp(float(f/c))-1
    return g


def logTransform(c, f):

    g = c * math.log(float(1 + f), 10)
    return g


def getLogLUT(value):

    c = 255 / math.log(float(1 + value), 10)

    lookUpTable = numpy.empty(256)
    for x in range(0, 256):
        lookUpTable[x] = round(logTransform(c, x))

    return lookUpTable



def getLogLUTForReverse(value):

    c = 255 / math.log(float(1 + value))
    
    loopUpTable = numpy.empty(256)
    for x in range(0,256):
        loopUpTable[x] = round(reverseTransform(c,x))

    return loopUpTable 


@RegisterAlgorithm("Log Operator", "PointwiseOp")
def logOperator(image):
  
    LUT = numpy.empty(256)
    LUT =  getLogLUT(numpy.max(image))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = LUT[image[i,j]]


    return {
        'processedImage': image
    }


@RegisterAlgorithm("Invert Operator", "PointwiseOp")
def invertLogOperator(image):
  
    LUT = numpy.empty(256)
    LUT =  getLogLUTForReverse(numpy.max(image))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = LUT[image[i,j]]


    return {
        'processedImage': image
    }

