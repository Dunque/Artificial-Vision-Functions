import numpy as np
from skimage import exposure

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