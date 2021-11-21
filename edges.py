import os
import numpy as np
from skimage import io

from filters import *

imgPathOut = "out/"

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
                #temp values used to represent the adyacent pixels
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