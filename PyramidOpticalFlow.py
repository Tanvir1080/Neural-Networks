import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib
from scipy import ndimage


""" This version of Lucas Kanade's optical flow algorithm utilizes the pyramid structure. 
    Each level of our pyramid cuts down the resolution of our image which allows for different forms of 
    feature tracking. 
    This lets you track features on both the large and small scale of an image. 
    The parameters for the pyramid are a window of (3,3) and 5 levels which have been found as the best parameters 
    while still maintaining our image. 

    The end result of this pyramid style is that features at the same location in the image across the different levels
    of the pyramid should have the vectors overlap however, there was not enough time to do this.  Therefore I am explaining 
    what the end result should be.  This means that at the end there should only be one image, insead of the multiple that I display here. 
    I am displayin multiple here to show that the pyramid style does work and i do generate corners and vectors that can be superimposed onto
    a single final image. 
"""
# to avoid errors when dividing 
np.seterr(divide='ignore', invalid='ignore')

def LucasKanade(img1, img2, currentLevel, numLevels):
    kernelSize = 7
    kernel = np.ones((kernelSize, kernelSize))/49

    height, width = img1.shape

    # find x and y gradients
    IxGradient = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    IyGradient = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)

    # since we only have 2 frames to look at the derivative with respect 
    # to time can be found by subtracting our frames
    ItGradient = img2 - img1

    # we blur our gradients to get proper corner features
    IxGradient = cv2.GaussianBlur(IxGradient, (kernelSize, kernelSize), 3)
    IyGradient = cv2.GaussianBlur(IyGradient, (kernelSize, kernelSize), 3)
    ItGradient = cv2.GaussianBlur(ItGradient, (kernelSize, kernelSize), 3)
    
    # with these we make our "tensors" as per the formula from the lecture slides
    Txx = cv2.filter2D(IxGradient**2, -1, kernel)
    Tyy = cv2.filter2D(IyGradient**2, -1, kernel)
    Txy = cv2.filter2D(IxGradient*IyGradient, -1, kernel)
    Txt = cv2.filter2D(IxGradient*ItGradient, -1, kernel)
    Tyt = cv2.filter2D(IyGradient*ItGradient, -1, kernel)

    denominator = cv2.filter2D((Txx*Tyy) - Txy**2, -1, kernel)
    
     # u and v hold the changes in distance in x and y of our corners
    u = ((Tyt*Txy)- (Txt*Tyy)) // denominator
    v = ((Txt*Txy)-(Tyt*Txx)) // denominator

    # find our "corners" that we want to track
    corners = cv2.goodFeaturesToTrack(img1, 100, 0.1, 10)
    corners = np.int0(corners)

    # so that we can add colors to our vector arrows
    kwargsParams = {
        "color" : ""
    }
    
    hsv = matplotlib.cm.get_cmap('hsv')

    # we loop through our corners to plot their changes on our image
    for i in corners:

        # get the x and y coordinates of our corners list 
        xCoordinate,yCoordinate = i.ravel()

        # cast them as ints to avoid plotting errors 
        xCoordinate = int(xCoordinate)
        yCoordinate = int(yCoordinate)

        # we change the color to match its direction and location
        kwargsParams["color"] = hsv((np.arctan2(xCoordinate, yCoordinate) + math.pi) / (2 * math.pi))

        # cast them as ints to avoid plotting errors 
        xCoordinate = int(xCoordinate)
        yCoordinate = int(yCoordinate)

        # to avoid errors because of unkown problems with how our corners are being accessed
        if(xCoordinate < height):
            plt.arrow(xCoordinate, yCoordinate, 5 * u[xCoordinate][yCoordinate], 5 * v[xCoordinate][yCoordinate], **kwargsParams)
        else:
            plt.arrow(xCoordinate, yCoordinate, 5 * u[yCoordinate][xCoordinate], 5 * v[yCoordinate][xCoordinate], **kwargsParams)

        # to help further identify the corner images we place circles at the start of the vector
        cv2.circle(img1,(xCoordinate,yCoordinate), 1, 255, -1)
    
    plt.imshow(img1, cmap="gray"), plt.title("Pyramid Level " + str(currentLevel)) ,plt.show()

def main():
    img1 = cv2.imread('basketball1.png', 0)
    img2 = cv2.imread('basketball2.png', 0)
    
    pyrLevels = 5
    windowSize = 3
    nLevel, img1Pyr = cv2.buildOpticalFlowPyramid(img1, (windowSize,windowSize), pyrLevels)
    nLevel, img2Pyr = cv2.buildOpticalFlowPyramid(img2, (windowSize,windowSize), pyrLevels)
    for i in range(nLevel + 1):
        LucasKanade(img1Pyr[2 * i], img2Pyr[2 * i], i, nLevel)

main()