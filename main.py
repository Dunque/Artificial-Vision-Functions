import os
import numpy as np
from natsort import natsorted, ns
from skimage.util import img_as_uint
from skimage.color import rgb2gray
from skimage import data, exposure, img_as_float, io, color
import matplotlib.pyplot as plt

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
# def filterImage (inImage, kernel):

#     #kernel cross correlation
#     #kernel = np.flipud(np.fliplr(kernel))

#     M = inImage.shape[0]
#     N = inImage.shape[1]

#     if (kernel.ndim == 1):
#         m = kernel.shape[0]
#         n = 1
#     else:
#         m = kernel.shape[0]
#         n = kernel.shape[1]

#     outImage = np.zeros((int(M - (m-1)), int(N - (n-1))))

#     for y in range(inImage.shape[1]):
#         if y > inImage.shape[1] - n:
#             break

#         for x in range(inImage.shape[0]):
#             if x > inImage.shape[0] - m:
#                 break
#             try:
#                 outImage[x, y] = (kernel * inImage[x: x + m, y: y + n]).sum()
#             except:
#                 break

#     return outImage

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
    result = [(1/(np.sqrt(2*np.pi)*sigma))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]  

    return result/sum(result)

#---------------------------------------------------------------------------

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()
 
    return kernel_2D
 
 
def gaussian_blur(inImage, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=np.sqrt(kernel_size), verbose=verbose)
    return filterImage(inImage, kernel)

# 3.3 -----------------------------------------------------------------
# 3.4 -----------------------------------------------------------------

def main():

    #Import files
    filename = os.path.join(imgPath, 'bfly.jpg')
    #bfly = io.imread(filename, as_gray=True)
    original = data.astronaut()
    bfly = rgb2gray(original)

    #bfly = adjustIntensity(bfly, [10,100], [0,1])

    #bfly = equalizeIntensity(bfly)

    kernel = np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                       [-1, -1, -1]])
    
    bfly = filterImage(bfly, kernel)

    #print(gaussKernel1D(2))

    #bfly = gaussianFilter(bfly, 3)

    #bfly = gaussian_blur(bfly, 5, False)

    bfly = bfly / bfly.max() #normalizes bfly in range 0 - 255
    bfly = 255 * bfly
    bfly = bfly.astype(np.uint8)
    io.imsave(os.path.join(imgPathOut, 'bflyOut.jpg'), bfly)


if __name__ =='__main__':
    main()
