import os
import re
import numpy as np
from natsort import natsorted, ns
from skimage.util import img_as_uint
from skimage.color import rgb2gray
from skimage import data, exposure, img_as_float, io, color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import math

imgPath = "img/"
imgPathOut = "out/"

# 3.1 -----------------------------------------------------------------
def adjustIntensity (inImage, inRange = [], outRange = [0, 1]):

    if (not inRange):
        imin = np.min(inImage)
        imax = np.max(inImage)
    else:
        imin = inRange[0]
        imax = inRange[1]

    omin = outRange[0]
    omax = outRange[1]

    inImage = np.clip(inImage, imin, imax)

    if imin != imax:
        inImage = (inImage - imin) / (imax - imin)
        return inImage * (omax - omin) + omin
    else:
        return np.clip(inImage, omin, omax)


def equalizeIntensity (inImage, nBins=256):

    hist, bin_centers = exposure.histogram(inImage, nBins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    out = np.interp(inImage.flat, bin_centers, img_cdf)

    return out.reshape(inImage.shape)

# 3.2 -----------------------------------------------------------------
def filterImage(inImage, kernel):
    
    M = inImage.shape[0]
    N = inImage.shape[1]

    if (kernel.ndim == 1):
        m = kernel.shape[0]
        n = 1
    else:
        m = kernel.shape[0]
        n = kernel.shape[1]
 
    output = np.zeros(inImage.shape)
 
    pad_height = int((m - 1) / 2)
    pad_width = int((n - 1) / 2)
 
    padded_image = np.zeros((M + (2 * pad_height), N + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = inImage
 
    for row in range(M):
        for col in range(N):
            output[row, col] = np.sum(kernel * padded_image[row:row + m, col:col + n])
 
    return output


def gaussKernel1D (sigma):
    N = int(2*np.ceil(3*sigma)+ 1)

    result = np.zeros(N)

    mid = int(N/2)
    result = np.array([(1/(np.sqrt(2*np.pi)*sigma))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)])  

    return result/sum(result)


def gaussianFilter (inImage, sigma):

    gausskernel = gaussKernel1D(sigma)

    image = filterImage(inImage, gausskernel)

    image2 = filterImage(image, np.transpose(gausskernel))

    return image2


def medianFilter (inImage, filterSize):

    indexer = filterSize // 2

    window = [
        (i, j)
        for i in range(-indexer, filterSize-indexer)
        for j in range(-indexer, filterSize-indexer)
    ]
    index = len(window) // 2

    for i in range(len(inImage)):
        for j in range(len(inImage[0])):
            inImage[i][j] = sorted(
                0 if (
                    min(i+a, j+b) < 0
                    or len(inImage) <= i+a
                    or len(inImage[0]) <= j+b
                ) else inImage[i+a][j+b]
                for a, b in window
            )[index]

    return inImage  


def highBoost (inImage, A, method, param):

    if method == "gaussian":
        smooth = gaussianFilter(inImage, param)
    elif method == "median":
        smooth = medianFilter(inImage, param)
    else:
        raise Exception("Invalid method, it should be gaussian or median")

    if A>=0:
        outImage = A * inImage - smooth
    else:
        outImage = (A-1) * inImage + inImage - smooth

    return outImage

# 3.3 -----------------------------------------------------------------

def erode(im,se):
    rows,columns = im.shape[0], im.shape[1]
    #Initialize counters (Just to keep track)
    fit = 0
    hit = 0
    miss = 0

    #Create a copy of the image to modified it´s pixel values
    ero = np.copy(im)
    #Specify kernel size (w*w)
    w = 3

    #
    for i in range(rows-w-1):
        for j in range(columns-w-1):
            #Get a region (crop) of the image equal to kernel size
            crop = im[i:w+i,j:w+j]
            #Convert region of image to an array
            img = np.array(crop)

            #Get center
            a = math.floor(w/2)
            b = math.floor(w/2)
            
            #Initialize counters 
            matches = 0
            blacks = 0

            #Count number of black pixels (0) and value matches between the two matrix
            for x in range(w):
                for y in range(w):
                    #Count number of black pixels (0)
                    if(img[x][y] == 0):
                        blacks = blacks+1
                        #Count number of matching pixel values between the two matrix   
                        if (img[x][y] == se[x][y]):         
                            matches = matches+1

            #Test if structuring element fit crop image pixels
            #If fit it does nothing because center pixel is already black
            if(matches > 0):
                if(matches == blacks):
                    #Touch
                    fit = fit + 1
                    pass
                #Test if structuring element hit crop image pixels
                #If hit change ero center pixel to black
                elif(matches < blacks):
                    #Hit
                    hit = hit+1
                    ##PROBABLE ERROR IN HERE##
                    ero[i+a][j+b] = 0
            #If no pixel match just pass
            else:
                #Miss
                miss=miss+1
                pass

    #Print the number of fits, hits, and misses
    print(str(fit) + '\n' + str(hit) + '\n' + str(miss))

    return ero

# 3.4 -----------------------------------------------------------------

def main():

    #Greyscale image
    filename = os.path.join(imgPath, 'bfly.jpeg')
    bfly = io.imread(filename, as_gray=True)

    #Black and white image
    filename = os.path.join(imgPath, 'bin.png')
    bw = io.imread(filename, as_gray=True)
    thresh = threshold_otsu(bw)
    bw = bw > thresh

    #bfly = adjustIntensity(bfly, [10,100], [0,1])

    #bfly = equalizeIntensity(bfly)

    kernel = np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                       [-1, -1, -1]])
    
    #bfly = filterImage(bfly, kernel)

    #bfly = gaussianFilter(bfly,1)

    #bfly = medianFilter(bfly,5)

    #bfly = highBoost(bfly, 1, "gaussian", 1)
    #bfly = highBoost(bfly, -1, "median", 1)

    se = [[0,1,0],
          [1,1,1],
          [0,1,0]]

    erode(bw,se)

    bfly = bfly / bfly.max()
    # io.imshow(bfly2, vmin=0, vmax=255) 
    # io.show()
    io.imsave(os.path.join(imgPathOut, 'bflyOut.jpg'), bfly)
    io.imsave(os.path.join(imgPathOut, 'bwOut.jpg'), bw)


if __name__ =='__main__':
    main()
