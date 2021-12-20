import numpy as np

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
        center = [int(np.floor(P/2)), int(np.floor(Q/2))]

    centerDiffRow = np.floor(P/2) - center[0]
    centerDiffCol = np.floor(Q/2) - center[1]

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
                try:
                    outImage[centeredRow,centeredCol] = 1
                except IndexError as e:
                    pass

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
        center = [int(np.floor(P/2)), int(np.floor(Q/2))]

    centerDiffRow = np.floor(P/2) - center[0]
    centerDiffCol = np.floor(Q/2) - center[1]

    outImage = np.zeros(inImage.shape)
 
    pad_height = int((P - 1) / 2)
    pad_width = int((Q - 1) / 2)
 
    padded_image = np.zeros((M + (2 * pad_height), N + (2 * pad_width)))

    #We copy the image to the padded one
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = inImage

    for row in range(M):
        for col in range(N):
            
            anyCheck = False

            crop = np.array(padded_image[row:P+row,col:Q+col], dtype=int)

            for row2 in range(crop.shape[0]):
                for col2 in range(crop.shape[1]):
                    if (SE[row2,col2] == 1 and crop[row2,col2] == 1):
                        anyCheck = True

            if (anyCheck):
                centeredRow = int(row + centerDiffRow)
                centeredCol = int(col + centerDiffCol)
                try:
                    outImage[centeredRow,centeredCol] = 1
                except IndexError as e:
                    pass

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

    outImage = np.zeros(inImage.shape)

    for row in range(M):
        for col in range(N):
            if out1[row,col] == out2[row,col]:
                outImage[row,col] = out1[row,col]

    return outImage
