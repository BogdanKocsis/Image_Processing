"""
Module docstring?
"""
import numpy
import math

from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.OutputDecorators import OutputDialog


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

    return lookUpTable.astype(int)



def getLogLUTForReverse(value):

    c = 255 / math.log(float(1 + value))
    
    loopUpTable = numpy.empty(256)
    for x in range(0,256):
        loopUpTable[x] = round(reverseTransform(c,x))

    return loopUpTable 


@RegisterAlgorithm("Log Operator", "PointwiseOp")
def logOperator(image):
  
    # LUT = numpy.empty(256)
    # LUT =  getLogLUT(numpy.max(image))


    # # for i in range(image.shape[0]):
    # #     for j in range(image.shape[1]):
    # #         image[i, j] = LUT[image[i,j]]

    c = 255 / numpy.log10(1 + numpy.max(image)) 
    log_image = c * (numpy.log10(image + 1)) 
   
    log_image = numpy.array(log_image, dtype = numpy.uint8)

    return {
        'processedImage': log_image
    }


@RegisterAlgorithm("Invert Operator", "PointwiseOp")
def invertLogOperator(image):
  
    # LUT = numpy.empty(256)
    # LUT =  getLogLUTForReverse(numpy.max(image))

    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         image[i, j] = LUT[image[i,j]]

    c = 255 / numpy.log(1 + numpy.max(image)) 
    log_image = numpy.exp(image/c)-1
    log_image = numpy.array(log_image, dtype = numpy.uint8)

    return {
        'processedImage': log_image
    }


@OutputDialog(title="Result")
@RegisterAlgorithm("Otsu's Binarization", "Binarization")
def otsu(image):
    if image.ndim == 2:
        # Image Properties
        histogram_array = numpy.histogram(image, bins=range(257), range=(-1, 255))[0]
        total_px_nr = numpy.prod(image.shape)
        colors_array = numpy.arange(256)

        # Probability
        prob_array = histogram_array / total_px_nr

        # Otsu Variables
        max_variance_btw_classes = float()
        optimal_threshold = int()
        
        # T
        threshold = 1
        while threshold != 255:
            class0 = prob_array[:threshold + 1]
            class1 = prob_array[threshold + 1:]
            p0 = numpy.sum(class0)
            p1 = numpy.sum(class1)
            mean0 = numpy.sum(prob_array[:threshold + 1] * colors_array[:threshold + 1]) / p0
            mean1 = numpy.sum(prob_array[threshold + 1:] * colors_array[threshold + 1:]) / p1
            variance_btw_classes = p0 * p1 * ((mean0 - mean1) ** 2)
            if variance_btw_classes > max_variance_btw_classes:
                max_variance_btw_classes = variance_btw_classes
                optimal_threshold = threshold
            threshold += 1

        image[image < optimal_threshold] = 0
        image[image >= optimal_threshold] = 255
        return {
            # 'processedImage': image,
            'outputMessage': "SUCCES\nThreshold: " + str(optimal_threshold)}
    else:
        return {
            'processedImage': "ERROR:\nImage isn't grayscale"}
