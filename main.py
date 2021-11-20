import os
import numpy as np
from skimage.color import rgb2gray
from skimage import data, exposure, io
from skimage.filters import threshold_otsu
import scipy

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

    #We copy the image to the padded one
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

    indexer = filterSize // 2

    window = [
        (i, j)
        for i in range(-indexer, filterSize-indexer)
        for j in range(-indexer, filterSize-indexer)
    ]
    index = len(window) // 2
 
    outImage = np.zeros(inImage.shape)
 
    for row in range(M):
        for col in range(N):
            outImage[row, col] = sorted(
                0 if (min(row+a, col+b) < 0 or len(inImage) <= row+a or len(inImage[0]) <= col+b) 
                else inImage[row+a][col+b] for a, b in window)[index]
 
    return outImage


def highBoost (inImage, A, method, param):

    if method == "gaussian":
        smooth = gaussianFilter(inImage, param)
    elif method == "median":
        smooth = medianFilter(inImage, param)
    else:
        raise Exception("Invalid method, it should be gaussian or median")

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
        raise "Error: given center is out of the SE"

    if (not center):
        center = [int(np.floor(P/2) + 1), int(np.floor(Q/2) + 1)]
    print(center)

    centerDiffRow = np.floor(P/2) + 1 - center[0]
    centerDiffCol = np.floor(Q/2) + 1 - center[1]

    print(centerDiffRow)
    print(centerDiffCol)

    outImage = np.zeros(inImage.shape)
 
    pad_height = int((P - 1) / 2)
    pad_width = int((Q - 1) / 2)
 
    padded_image = np.zeros((M + (2 * pad_height), N + (2 * pad_width)))

    #We copy the image to the padded one
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = inImage

    for row in range(M):
        for col in range(N):

            crop = np.array(padded_image[row:P+row,col:Q+col])

            if (crop.all() == SE.all()):
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
        raise "Error: given center is out of the SE"

    if (not center):
        center = [int(np.floor(P/2) + 1), int(np.floor(Q/2) + 1)]
    print(center)

    centerDiffRow = np.floor(P/2) + 1 - center[0]
    centerDiffCol = np.floor(Q/2) + 1 - center[1]

    print(centerDiffRow)
    print(centerDiffCol)

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

    if(objSE.len != bgSE.len or objSE.any() == bgSE.any()):
        raise "Error: objSE and bgSE must not share ones in the same positions"

    out1 = erode (B, objSE, center)
    out2 = erode (Bc, bgSE, center)

    outInd = np.intersect1d(out1, out2, return_indices=True)

    outImage = np.zeros(inImage.shape)

    for i in outInd:
        outImage[i] = 1

    return outImage

# 3.4 -----------------------------------------------------------------

def gradientImage (inImage, operator):

    if (operator == "Roberts"):

        Gx = np.array([[-1, 0], 
                       [ 0, 1]])

        Gy = np.array([[ 0, -1], 
                       [ 1, 0 ]])

        # io.imsave(os.path.join(imgPathOut, 'gx.jpg'), filterImage(inImage, Gx))
        # io.imsave(os.path.join(imgPathOut, 'gy.jpg'), filterImage(inImage, Gy))
        
        return [filterImage(inImage, Gx) , filterImage(inImage, Gy)]

    elif (operator == "CentralDiff"):

        Gx = np.array([-1, 0, 1])

        Gy = Gx.transpose()

        # io.imsave(os.path.join(imgPathOut, 'gx.jpg'), filterImage(inImage, Gx))
        # io.imsave(os.path.join(imgPathOut, 'gy.jpg'), filterImage(inImage, Gy))
        
        return [filterImage(inImage, Gx) , filterImage(inImage, Gy)]

    elif (operator == "Prewitt"):

        Gx = np.array([[-1, 0, 1], 
                       [-1, 0, 1], 
                       [-1, 0, 1]])

        Gy = np.array([[-1, -1, -1], 
                       [ 0,  0,  0], 
                       [ 1,  1,  1]])

        # io.imsave(os.path.join(imgPathOut, 'gx.jpg'), filterImage(inImage, Gx))
        # io.imsave(os.path.join(imgPathOut, 'gy.jpg'), filterImage(inImage, Gy))
        
        return [filterImage(inImage, Gx) , filterImage(inImage, Gy)]

    elif (operator == "Sobel"):

        Gx = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])

        Gy = np.array([[-1, -2, -1], 
                       [ 0,  0,  0], 
                       [ 1,  2,  1]])

        # io.imsave(os.path.join(imgPathOut, 'gx.jpg'), filterImage(inImage, Gx))
        # io.imsave(os.path.join(imgPathOut, 'gy.jpg'), filterImage(inImage, Gy))
        
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
    filename = os.path.join(imgPath, 'bfly.jpeg')
    #bfly = io.imread(filename, as_gray=True)
    bfly = rgb2gray(data.astronaut())

    #Black and white image
    filename = os.path.join(imgPath, 'bw.jpg')
    bw = io.imread(filename, as_gray=True)
    thresh = threshold_otsu(bw)
    bw = bw > thresh
    bw = bw * 1

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

    # se = np.array([[1,1,1],
    #                [1,1,1],
    #                [1,1,1]])

    se = np.array([[1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1]])
    
    center = [6,6]

    #bw = erode(bw,se,center)

    #bw = dilate(bw,se,center)

    #bfly = gradientImage (bfly, "Roberts")
    #bfly = gradientImage (bfly, "CentralDiff")
    #bfly = gradientImage (bfly, "Prewitt")
    #bfly = gradientImage (bfly, "Sobel")

    # bfly = edgeCanny(bfly, 0.4, 25, 255)
    
    io.imsave(os.path.join(imgPathOut, 'sciConv.jpg'), scipy.signal.convolve(bfly, np.array([-1, 0, 1]), mode='full', method='direct'))
    io.imsave(os.path.join(imgPathOut, 'myConv.jpg'), filterImage(bfly, np.array([-1, 0, 1])))
    #bfly = bfly / bfly.max()
    io.imsave(os.path.join(imgPathOut, 'bflyOut.jpg'), bfly)
    #io.imsave(os.path.join(imgPathOut, 'bwOut.jpg'), bw)


if __name__ =='__main__':
    main()
