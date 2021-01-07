"""
Module docstring?
"""
import numpy
import collections
import math
import skimage
from skimage.filters import threshold_otsu

from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.OutputDecorators import OutputDialog
from Application.Utils.InputDecorators import InputDialog


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
    for x in range(0, 256):
        loopUpTable[x] = round(reverseTransform(c, x))

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

    log_image = numpy.array(log_image, dtype=numpy.uint8)

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
    log_image = numpy.array(log_image, dtype=numpy.uint8)

    return {
        'processedImage': log_image
    }


@OutputDialog(title="Result")
@RegisterAlgorithm("Otsu's Binarization", "Binarization")
def otsu(image):
    if image.ndim == 2:

        histogram_array = numpy.histogram(
            image, bins=range(257), range=(-1, 256))[0]
        total_px_nr = numpy.prod(image.shape)
        k = numpy.arange(256)

        prob_array = histogram_array / total_px_nr

        max_variance_btw_classes = float()
        optimal_threshold = int()

        threshold = 1
        while threshold != 255:
            class1 = prob_array[:threshold + 1]
            class2 = prob_array[threshold + 1:]
            p1 = numpy.sum(class1)
            p2 = numpy.sum(class2)
            mean1 = numpy.sum(
                prob_array[:threshold + 1] * k[:threshold + 1]) / p1
            mean2 = numpy.sum(
                prob_array[threshold + 1:] * k[threshold + 1:]) / p2
            variance_btw_classes = p1 * p2 * ((mean1 - mean2) ** 2)
            if variance_btw_classes > max_variance_btw_classes:
                max_variance_btw_classes = variance_btw_classes
                optimal_threshold = threshold
            threshold += 1

        image[image < optimal_threshold] = 0
        image[image >= optimal_threshold] = 255
        return {
            'processedImage': image,
            # 'outputMessage': "SUCCES\nThreshold: " + str(optimal_threshold)
        }
    else:
        return {
            'processedImage': "ERROR:\nImage isn't grayscale"}


def computeIntegralImage(image):

    intImageArray = numpy.zeros([image.shape[0], image.shape[1]], dtype=int)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if(j != 0):
                intImageArray[i, j] = intImageArray[i, j-1] + image[i, j]
            else:
                intImageArray[i, j] = image[i, j]

    for i in range(1, image.shape[0]):
        for j in range(0, image.shape[1]):
            intImageArray[i, j] = intImageArray[i-1, j] + intImageArray[i, j]

    return intImageArray


@RegisterAlgorithm("Mean Filter", "Filter")
@InputDialog(maskSize=int)
def meanFilter(image, maskSize=3):
    if image.ndim == 2:
        if maskSize % 2 == 0:
            maskSize += 1

        intImageArray = computeIntegralImage(image)

        filterPadding = maskSize // 2
        print(filterPadding)

        for i in range(filterPadding + 1, intImageArray.shape[0] - filterPadding - 1):
            for j in range(filterPadding + 1, intImageArray.shape[1] - filterPadding - 1):
                _sum =\
                    intImageArray[i+filterPadding][j+filterPadding] +\
                    intImageArray[i-filterPadding-1][j-filterPadding-1] -\
                    intImageArray[i+filterPadding][j-filterPadding-1] -\
                    intImageArray[i-filterPadding-1][j+filterPadding]
                mean = _sum / (maskSize**2)
                image[i, j] = mean

        filteredImage = numpy.array(image, dtype=numpy.uint8)

        return {
            'processedImage': filteredImage
        }
    else:
        return {
            'processedImage': "ERROR:\nImage isn't grayscale"}

@InputDialog(threshold=int)
@OutputDialog(title="Result")
@RegisterAlgorithm("Sobel", "Filter")
def sobel_filter(image, threshold):
    if image.ndim == 2:
        image_width = image.shape[1]
        image_height = image.shape[0]
        target_image = numpy.empty([image.shape[0], image.shape[1]])

        h1 = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        h2 = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        for i in range(len(image)):
            for j in range(len(image[i])):
                if i == 0 or j == 0 or i == image_height - 1 or j == image_width - 1:
                    target_image[i][j] = 0
                else:
                    this_slice = image[i - 1:i + 2, j - 1:j + 2]
                    g1 = this_slice * h1
                    g2 = this_slice * h2
                    g1_sum = numpy.sum(g1)
                    g2_sum = numpy.sum(g2)
                    g = math.sqrt((g1_sum ** 2) + (g2_sum ** 2))

                    target_image[i][j] = 255 if g >= threshold else 0
        return {
            'processedImage': target_image.astype(numpy.uint8),
            'outputMessage': "SUCCESS"}

    else:
        return {
            'outputMessage': "ERROR:\nImage isn't grayscale"}


@InputDialog(threshold=int)
@RegisterAlgorithm("Sobel Vertical", "Filter")
def sobel_filter_vertical(image, threshold):
    if image.ndim == 2:
        image_width = image.shape[1]
        image_height = image.shape[0]
        target_image = numpy.empty([image.shape[0], image.shape[1]])

        for i in range(1, image_height-1):
            for j in range(1, image_width-1):
                Fy = int(image[i-1, j+1])-int(image[i-1, j-1])+2*int(image[i, j+1]) - \
                    2*int(image[i, j-1])+int(image[i+1, j+1]) - \
                    int(image[i+1, j-1])
                Fx = int(image[i+1, j-1])-int(image[i-1, j-1])+2*int(image[i+1, j]) - \
                    2*int(image[i-1, j])+int(image[i+1, j+1]) - \
                    int(image[i-1, j+1])
                g = math.sqrt((Fx ** 2) + (Fy ** 2))

                if g >= threshold:
                    theta = math.atan2(Fx, Fy) * (180 / math.pi)
                    target_image[i, j] = 255 if (theta >= -5 and theta <= 5) or (
                        theta >= -180 and theta <= -175) or (theta >= 175 and theta <= 180) else 0

        target_image = numpy.array(target_image, dtype=numpy.uint8)
        return {
            'processedImage': target_image
        }

    else:
        return {
            'outputMessage': "ERROR:\nImage isn't grayscale"}



def erosion(image, maskSize=3):

    target_image = numpy.empty([image.shape[0], image.shape[1]])
    img_height = image.shape[0]
    img_width = image.shape[1]
    border = maskSize // 2

    for y in range(border, img_height - border):
        for x in range(border, img_width - border):
            blackPixel = False
            for i in range(-border, border+1):
                for j in range(-border, border+1):
                    if int(image[y+i, x+j]) == 0:
                        blackPixel = True
                        break

            oldGray = int(image[y, x])

            if oldGray == 0 or blackPixel:
                target_image[y, x] = 0
            else:
                target_image[y, x] = 255

    return target_image.astype(numpy.uint8)


def dilatation(image, maskSize=3):
    target_image = numpy.empty([image.shape[0], image.shape[1]])
    img_height = image.shape[0]
    img_width = image.shape[1]
    border = maskSize // 2

    for y in range(border, img_height - border):
        for x in range(border, img_width - border):
            whitePixel = False
            for i in range(-border, border+1):
                for j in range(-border, border+1):
                    if int(image[y+i, x+j]) == 255:
                        whitePixel = True
                        break

            oldGray = int(image[y, x])

            if oldGray == 255 or whitePixel:
                target_image[y, x] = 255
            else:
                target_image[y, x] = 0

    return target_image.astype(numpy.uint8)


@RegisterAlgorithm("Opening", "Morphology")
@InputDialog(maskSize=int)
@OutputDialog(title="Result")
def opening(image, maskSize=3):
    histogram_array = numpy.histogram(
        image, bins=range(257), range=(-1, 255))[0]
    boolean_histogram_array = histogram_array != 0
    if numpy.any(boolean_histogram_array[1:255]):
        return {
            'outputMessage': "ERROR:\nImage isn't binarized"}

    image_eroded = erosion(image, maskSize)
    image_result = dilatation(image_eroded, maskSize)

    return {
        'processedImage': image_result.astype(numpy.uint8),
    }

def bilinearRotation(image, degrees):

    # Image
    image_width = image.shape[1]
    image_height = image.shape[0]
    target_image = numpy.zeros(
        [image.shape[0], image.shape[1]], dtype=numpy.uint8)
    radians = degrees * math.pi/180
    center_x = image_width / 2
    center_y = image_height / 2

    for y_1 in range(0, image_height):
        for x_1 in range(0, image_width):

            xc = (x_1-center_x) * math.cos(radians) - \
                (y_1-center_y)*math.sin(radians) + center_x
            yc = (x_1-center_x) * math.sin(radians) + \
                (y_1-center_y)*math.cos(radians) + center_y

            x = int(math.floor(xc))
            y = int(math.floor(yc))

            xd = float(xc - int(xc))
            yd = float(yc - int(yc))
            if (xc >= 0) and (xc < target_image.shape[1] - 1) and (yc >= 0) and (yc < target_image.shape[0] - 1):
                target_image[y_1, x_1] = (1 - xd) * (1 - yd) * int(image[y, x]) + xd * (1 - yd) * int(image[y, x + 1]) + (
                    1 - xd) * yd * int(image[y + 1, x]) + xd * yd * int(image[y + 1, x + 1])

    return numpy.round_(target_image).astype(numpy.uint8)




@RegisterAlgorithm("Rotation", "Interpolation")
@InputDialog(degrees=int)
@OutputDialog(title="Result")
def rotation(image, degrees):
    if image.ndim == 2:

        target_image = bilinearRotation(image, degrees)
        index = 1
        while index < 4:
            target_image = bilinearRotation(target_image, degrees)
            index= index + 1

        return {
            'processedImage': numpy.round_(target_image).astype(numpy.uint8)}
    else:
        return {
            'outputMessage': "ERROR:\nImage isn't grayscale"}

@RegisterAlgorithm("HoughTransform", "Hough")
@OutputDialog(title="Result")
def houghTransform(image):
    # Binarization Test
    histogram_array = numpy.histogram(image, bins=range(257), range=(-1, 255))[0]
    boolean_histogram_array = histogram_array != 0
    if numpy.any(boolean_histogram_array[1:255]):
        return {
            'outputMessage': "ERROR:\nImage isn't binarized"}
    else :

        image_width = image.shape[1]
        image_height = image.shape[0]
        hough_matrix = numpy.zeros([int(math.sqrt((image_height ** 2) + (image_width ** 2))) + 2, 271], dtype=numpy.int)

        white_px_index = numpy.where( image == 255)
        white_px_coords = list(zip(white_px_index[0], white_px_index[1]))
        for px in white_px_coords:
            for alpha in range(-90, 180):
                radius = math.cos(alpha * math.pi/180) * px[1] + math.sin(alpha * math.pi/180)*px[0]
                if radius >= 0:
                    hough_matrix[numpy.int(radius),alpha+90] +=1

        h_max = numpy.max(hough_matrix)
        hough_matrix_scaled = 255.0 / h_max * hough_matrix

        return {
            'processedImage':  hough_matrix_scaled.astype(numpy.uint8)

        }

