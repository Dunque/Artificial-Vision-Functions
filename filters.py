
import numpy as np
from sklearn.preprocessing import minmax_scale

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