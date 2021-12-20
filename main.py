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
    filename = os.path.join(imgPath, 'circles.png')
    #img1 = io.imread(filename, as_gray=True)
    img1 = rgb2gray(data.astronaut())

    filename2 = os.path.join(imgPath, 'gaussianFilter.png')
    ker = io.imread(filename2, as_gray=True)
    ker = rgb2gray(ker)

    #Black and white image
    filename = os.path.join(imgPath, 'morph.png')
    bw2 = io.imread(filename, as_gray=True)
    thresh = threshold_otsu(bw2)
    bw2 = bw2 > thresh
    bw2 = bw2 * 1

    # 3.1 -----------------------------------------------------------------


    #io.imsave(os.path.join(imgPathOut, 'adjustIntensity.png'), adjustIntensity(img1, [0.1,0.9], [0,1]))

    #io.imsave(os.path.join(imgPathOut, 'equalizeIntensity.png'), equalizeIntensity(img1,4))


    # 3.2 -----------------------------------------------------------------


    kernel = np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                       [-1, -1, -1]])

    #io.imsave(os.path.join(imgPathOut, 'filterImage.png'), filterImage(img1, kernel))

    #io.imsave(os.path.join(imgPathOut, 'gaussianFilter.png'), gaussianFilter(img1,1.25))

    #io.imsave(os.path.join(imgPathOut, 'medianFilter.png'), medianFilter(img1,15))

    #io.imsave(os.path.join(imgPathOut, 'highBoostGaussian.png'), highBoost(ker, 2, "gaussian", 1.75))
    #io.imsave(os.path.join(imgPathOut, 'highBoostMedian.png'), highBoost(img1, 5, "median", 3))


    # 3.3 -----------------------------------------------------------------


    se = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    
    center = [1,1]

    #io.imsave(os.path.join(imgPathOut, 'erode.png'), erode(bw2,se,center))

    #io.imsave(os.path.join(imgPathOut, 'dilate.png'), dilate(bw2,se,center))

    #io.imsave(os.path.join(imgPathOut, 'opening.png'), opening(bw2,se,center))

    #io.imsave(os.path.join(imgPathOut, 'closing.png'), closing(bw2,se,center))

    objSE = np.array([[0,1,0],
                      [0,1,1],
                      [0,1,0]])

    bgSE = np.array([[1,0,1],
                     [0,0,0],
                     [1,0,1]])


    #io.imsave(os.path.join(imgPathOut, 'HitOrMiss.png'), hit_or_miss(bw2,objSE,bgSE))


    # 3.4 -----------------------------------------------------------------


    #gradientImage (img1, "Roberts")
    #gradientImage (img1, "CentralDiff")
    #gradientImage (img1, "Prewitt")
    #gradientImage (img1, "Sobel")
    
    io.imsave(os.path.join(imgPathOut, 'edgeCanny.png'), edgeCanny(img1, 0.4, 25, 100))


if __name__ =='__main__':
    main()