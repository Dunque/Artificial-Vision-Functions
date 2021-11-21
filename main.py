import os
import numpy as np
from skimage.color import rgb2gray
from skimage import data, io
from skimage.filters import threshold_otsu

from histograms import *
from morphology import *
from filters import *
from edges import *

imgPath = "img/"
imgPathOut = "out/"

def main():

    #Greyscale image
    filename = os.path.join(imgPath, 'img1.jpeg')
    #img1 = io.imread(filename, as_gray=True)
    img1 = rgb2gray(data.astronaut())

    #Black and white image
    filename = os.path.join(imgPath, 'input92.png')
    bw2 = io.imread(filename, as_gray=True)
    thresh = threshold_otsu(bw2)
    bw2 = bw2 > thresh
    bw2 = bw2 * 1

    # 3.1 -----------------------------------------------------------------


    io.imsave(os.path.join(imgPathOut, 'adjustIntensity.jpg'), adjustIntensity(img1, [0.1,0.9], [0,1]))

    io.imsave(os.path.join(imgPathOut, 'equalizeIntensity.jpg'), equalizeIntensity(img1))


    # 3.2 -----------------------------------------------------------------


    kernel = np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                       [-1, -1, -1]])

    io.imsave(os.path.join(imgPathOut, 'filterImage.jpg'), filterImage(img1, kernel))

    io.imsave(os.path.join(imgPathOut, 'gaussianFilter.jpg'), gaussianFilter(img1,1))

    io.imsave(os.path.join(imgPathOut, 'medianFilter.jpg'), medianFilter(img1,5))

    io.imsave(os.path.join(imgPathOut, 'highBoostGaussian.jpg'), highBoost(img1, 5, "gaussian", 1))
    io.imsave(os.path.join(imgPathOut, 'highBoostMedian.jpg'), highBoost(img1, 5, "median", 3))


    # 3.3 -----------------------------------------------------------------


    se = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    
    center = [1,1]

    io.imsave(os.path.join(imgPathOut, 'erode.jpg'), erode(bw2,se,center))

    io.imsave(os.path.join(imgPathOut, 'dilate.jpg'), dilate(bw2,se,center))

    io.imsave(os.path.join(imgPathOut, 'opening.jpg'), opening(bw2,se,center))

    io.imsave(os.path.join(imgPathOut, 'closing.jpg'), closing(bw2,se,center))

    objSE = np.array([[0,0,0],
                      [1,1,0],
                      [0,1,0]])

    bgSE = np.array([[0,1,1],
                     [0,0,1],
                     [0,0,0]])


    io.imsave(os.path.join(imgPathOut, 'HitOrMiss.jpg'), hit_or_miss(bw2,objSE,bgSE))


    # 3.4 -----------------------------------------------------------------


    gradientImage (img1, "Roberts")
    gradientImage (img1, "CentralDiff")
    gradientImage (img1, "Prewitt")
    gradientImage (img1, "Sobel")
    
    io.imsave(os.path.join(imgPathOut, 'edgeCanny.jpg'), edgeCanny(img1, 0.4, 25, 255))


if __name__ =='__main__':
    main()