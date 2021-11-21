import os
import numpy as np
from sklearn.preprocessing import minmax_scale
from skimage.color import rgb2gray
from skimage import data, exposure, io, morphology
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity

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
 
    outImage = np.zeros(inImage.shape)
 
    pad_height = int((m - 1) / 2)
    pad_width = int((n - 1) / 2)
 
    padded_image = np.zeros((M + (2 * pad_height), N + (2 * pad_width)))

    #Copy the image to the padded one
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = inImage
 
    for row in range(M):
        for col in range(N):
            outImage[row, col] = np.sum(kernel * padded_image[row:row + m, col:col + n])
 
    return outImage


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
        
    M = inImage.shape[0]
    N = inImage.shape[1]

    center = [int(np.floor(filterSize/2)), int(np.floor(filterSize/2))]

    if (filterSize // 2 == 0):
        centerUp = filterSize-center[1]
        centerDown = filterSize-center[1]-1
        centerL = filterSize-center[0]
        centerR = filterSize-center[0]-1
    else:
        centerUp = filterSize-center[1]
        centerDown = filterSize-center[1]
        centerL = filterSize-center[0]
        centerR = filterSize-center[0]
 
    outImage = np.zeros(inImage.shape)
 
    for row in range(M):
        for col in range(N):
            pixels = []
            for i in range(row-centerL, row+centerR):
                for j in range(col-centerUp, col+centerDown):

                    try:
                        pixels.append(inImage[i,j])

                    except IndexError as e:
                        pass

            outImage[row, col] = np.median(pixels)
            
    return outImage


def highBoost (inImage, A, method, param):

    if method == "gaussian":
        smooth = gaussianFilter(inImage, param)
    elif method == "median":
        smooth = medianFilter(inImage, param)
    else:
        raise Exception("Invalid method, it should be gaussian or median")

    smooth = minmax_scale(smooth, feature_range=(0, 1), axis=0, copy=True)

    outImage = (A-1) * inImage + inImage - smooth

    return outImage

# 3.3 -----------------------------------------------------------------

def erode (inImage, SE, center=[]):

    M = inImage.shape[0]
    N = inImage.shape[1]
    
    if (SE.ndim == 1):
        P = SE.shape[0]
        Q = 1
    else:
        P = SE.shape[0]
        Q = SE.shape[1]

    if (center and (center[0] > P or center[1] > Q)):
        raise Exception("Given center is out of the SE")

    if (not center):
        center = [int(np.floor(P/2) + 1), int(np.floor(Q/2) + 1)]

    centerDiffRow = np.floor(P/2) + 1 - center[0]
    centerDiffCol = np.floor(Q/2) + 1 - center[1]

    outImage = np.zeros(inImage.shape,dtype=int)
 
    pad_height = int((P - 1) / 2)
    pad_width = int((Q - 1) / 2)
 
    padded_image = np.zeros((M + (2 * pad_height), N + (2 * pad_width)),dtype=int)

    #We copy the image to the padded one
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = inImage

    for row in range(M):
        for col in range(N):

            allCheck = True

            crop = np.array(padded_image[row:P+row,col:Q+col], dtype=int)

            for row2 in range(crop.shape[0]):
                for col2 in range(crop.shape[1]):
                    if (SE[row2,col2] == 1):
                        if(crop[row2,col2] != 1):
                            allCheck = False

            if (allCheck):
                centeredRow = int(row + centerDiffRow)
                centeredCol = int(col + centerDiffCol)
                outImage[centeredRow,centeredCol] = 1

    return outImage


def dilate (inImage, SE, center=[]):

    M = inImage.shape[0]
    N = inImage.shape[1]
    
    if (SE.ndim == 1):
        P = SE.shape[0]
        Q = 1
    else:
        P = SE.shape[0]
        Q = SE.shape[1]

    if (center and (center[0] > P or center[1] > Q)):
        raise Exception("Given center is out of the SE")

    if (not center):
        center = [int(np.floor(P/2) + 1), int(np.floor(Q/2) + 1)]

    centerDiffRow = np.floor(P/2) + 1 - center[0]
    centerDiffCol = np.floor(Q/2) + 1 - center[1]

    outImage = np.zeros(inImage.shape)
 
    pad_height = int((P - 1) / 2)
    pad_width = int((Q - 1) / 2)
 
    padded_image = np.zeros((M + (2 * pad_height), N + (2 * pad_width)))

    #We copy the image to the padded one
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = inImage

    for row in range(M):
        for col in range(N):

            crop = np.array(padded_image[row:P+row,col:Q+col])
            if (crop[center[0],center[1]] == SE[center[0],center[1]]):
                centeredRow = int(row + centerDiffRow)
                centeredCol = int(col + centerDiffCol)
                outImage[centeredRow:P+centeredRow,centeredCol:Q+centeredCol] = 1

    return outImage


def opening (inImage, SE, center=[]):

    inImage1 = erode (inImage, SE, center)

    outImage = dilate (inImage1, SE, center)

    return outImage


def closing (inImage, SE, center=[]):

    inImage1 = dilate (inImage, SE, center)

    outImage = erode (inImage1, SE, center)

    return outImage


def hit_or_miss (inImage, objSE, bgSE, center=[]):

    B = inImage
    Bc = 1 - inImage

    M = inImage.shape[0]
    N = inImage.shape[1]
    
    if(objSE.shape != bgSE.shape):
        raise Exception("objSE and bgSE must not share ones in the same positions")

    for row in range(objSE.shape[0]):
        for col in range(objSE.shape[1]):
            if objSE[row,col] == bgSE[row,col] == 1:
                raise Exception("objSE and bgSE must not share ones in the same positions")

    out1 = erode (B, objSE, center)
    out2 = erode (Bc, bgSE, center)

    # out1 = morphology.erosion(B, objSE)
    # out2 = morphology.erosion(Bc, bgSE)

    io.imsave(os.path.join(imgPathOut, 'out1BIEN.jpg'), morphology.erosion(B, objSE))
    io.imsave(os.path.join(imgPathOut, 'out2BIEN.jpg'), morphology.erosion(Bc, bgSE))
    io.imsave(os.path.join(imgPathOut, 'out1.jpg'), out1)
    io.imsave(os.path.join(imgPathOut, 'out2.jpg'), out2)

    outImage = np.zeros(inImage.shape)

    for row in range(M):
        for col in range(N):
            if out1[row,col] == out2[row,col]:
                outImage[row,col] = out1[row,col]

    return outImage


# 3.4 -----------------------------------------------------------------


def gradientImage (inImage, operator):

    if (operator == "Roberts"):

        Gx = np.array([[-1, 0], 
                       [ 0, 1]])

        Gy = np.array([[ 0, -1], 
                       [ 1, 0 ]])

        io.imsave(os.path.join(imgPathOut, 'robertsGx.jpg'), filterImage(inImage, Gx))
        io.imsave(os.path.join(imgPathOut, 'robertsGy.jpg'), filterImage(inImage, Gy))
        
        return [filterImage(inImage, Gx) , filterImage(inImage, Gy)]

    elif (operator == "CentralDiff"):

        Gx = np.array([-1, 0, 1])

        Gy = Gx.transpose()

        io.imsave(os.path.join(imgPathOut, 'centralDiffGx.jpg'), filterImage(inImage, Gx))
        io.imsave(os.path.join(imgPathOut, 'centralDiffGy.jpg'), filterImage(inImage, Gy))
        
        return [filterImage(inImage, Gx) , filterImage(inImage, Gy)]

    elif (operator == "Prewitt"):

        Gx = np.array([[-1, 0, 1], 
                       [-1, 0, 1], 
                       [-1, 0, 1]])

        Gy = np.array([[-1, -1, -1], 
                       [ 0,  0,  0], 
                       [ 1,  1,  1]])

        io.imsave(os.path.join(imgPathOut, 'prewittGx.jpg'), filterImage(inImage, Gx))
        io.imsave(os.path.join(imgPathOut, 'prewittGy.jpg'), filterImage(inImage, Gy))
        
        return [filterImage(inImage, Gx) , filterImage(inImage, Gy)]

    elif (operator == "Sobel"):

        Gx = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])

        Gy = np.array([[-1, -2, -1], 
                       [ 0,  0,  0], 
                       [ 1,  2,  1]])

        io.imsave(os.path.join(imgPathOut, 'sobelGx.jpg'), filterImage(inImage, Gx))
        io.imsave(os.path.join(imgPathOut, 'sobelGy.jpg'), filterImage(inImage, Gy))
        
        return [filterImage(inImage, Gx) , filterImage(inImage, Gy)]

    else:
        raise Exception("Invalid method, it should be gaussian or median")


def edgeCanny (inImage, sigma, tlow, thigh):

    #Noise Reduction

    outImage = gaussianFilter (inImage, sigma)

    #Gradient

    [Jx,Jy] = gradientImage (outImage, "Sobel")

    #Suppression

    M = outImage.shape[0]
    N = outImage.shape[1]
 
    Em = np.zeros(inImage.shape)
    Eo = np.zeros(inImage.shape)
 
    for row in range(M):
        for col in range(N):
            Em[row, col] = np.sqrt( pow(Jx[row, col],2) + pow(Jy[row, col],2) )
    
    io.imsave(os.path.join(imgPathOut, 'em.jpg'), Em)
    Em = Em / Em.max() * 255
    Eo = np.arctan2( Jy, Jx )

    #Threshold

    outImage3 = np.zeros(inImage.shape)

    #arctan values to angles conversion
    angle = Eo * 180. / np.pi
    angle[angle < 0] += 180

    for row in range(M):
        for col in range(N):
            try:
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[row,col] < 22.5) or (157.5 <= angle[row,col] <= 180):
                    q = Em[row, col+1]
                    r = Em[row, col-1]
                #angle 45
                elif (22.5 <= angle[row,col] < 67.5):
                    q = Em[row+1, col-1]
                    r = Em[row-1, col+1]
                #angle 90
                elif (67.5 <= angle[row,col] < 112.5):
                    q = Em[row+1, col]
                    r = Em[row-1, col]
                #angle 135
                elif (112.5 <= angle[row,col] < 157.5):
                    q = Em[row-1, col-1]
                    r = Em[row+1, col+1]

                if (Em[row,col] >= q) and (Em[row,col] >= r):
                    outImage3[row,col] = Em[row,col]
                else:
                    outImage3[row,col] = 0

            except IndexError as e:
                pass

    #Hysteresis

    for row in range(M):
        for col in range(N):
            if (outImage3[row,col] <= tlow):
                try:
                    if ((outImage3[row+1, col-1] <= thigh) or (outImage3[row+1, col] <= thigh) or (outImage3[row+1, col+1] <= thigh)
                        or (outImage3[row, col-1] <= thigh) or (outImage3[row, col+1] <= thigh)
                        or (outImage3[row-1, col-1] <= thigh) or (outImage3[row-1, col] <= thigh) or (outImage3[row-1, col+1] <= thigh)):
                        outImage3[row, col] = thigh
                    else:
                        outImage3[row, col] = 0

                except IndexError as e:
                    pass

    return 255 - outImage3


# ---------------------------------------------------------------------


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

    io.imsave(os.path.join(imgPathOut, 'adjustIntensity.jpg'), adjustIntensity(img1, [0.1,0.9], [0,1]))

    io.imsave(os.path.join(imgPathOut, 'equalizeIntensity.jpg'), equalizeIntensity(img1))

    kernel = np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                       [-1, -1, -1]])

    io.imsave(os.path.join(imgPathOut, 'filterImage.jpg'), filterImage(img1, kernel))

    io.imsave(os.path.join(imgPathOut, 'gaussianFilter.jpg'), gaussianFilter(img1,1))

    io.imsave(os.path.join(imgPathOut, 'medianFilter.jpg'), medianFilter(img1,5))

    io.imsave(os.path.join(imgPathOut, 'highBoostGaussian.jpg'), highBoost(img1, 5, "gaussian", 1))
    io.imsave(os.path.join(imgPathOut, 'highBoostMedian.jpg'), highBoost(img1, 5, "median", 3))

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

    gradientImage (img1, "Roberts")
    gradientImage (img1, "CentralDiff")
    gradientImage (img1, "Prewitt")
    gradientImage (img1, "Sobel")
    
    io.imsave(os.path.join(imgPathOut, 'edgeCanny.jpg'), edgeCanny(img1, 0.4, 25, 255))


if __name__ =='__main__':
    main()