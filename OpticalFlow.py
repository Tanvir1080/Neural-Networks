import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib
from scipy import ndimage

""" Optical flow is used to track specific features of an image as they change location across multiple frames of a scene.
    This can be used in many different fields as its a basic form of Object Tracking.  The Lucas Kanade method is used very often and 
    It takes an X and Y gradient of the original image along with a Time gradient which can be calculated by subtracting the intensities across
    the two images. After testing the kernel size of 7 seemed to be the best as it allows for just the proper amount of blurring to achieve proper 
    mathematical calculations when we make our "tensors".  These are then used to calculate our u and v which hold directionsl changes in the X and Y 
    for all pixels.  We then track through our corners and plot the displacement of those values using u and v to calculate the optical flow paths. """

# to avoid errors when dividing 
np.seterr(divide='ignore', invalid='ignore')

def LucasKanade(img1, img2):
    kernelSize = 7
    kernel = np.ones((kernelSize, kernelSize))/49

    height, width = img1.shape

    # find x and y gradients
    IxGradient = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    IyGradient = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)

    # since we only have 2 frames to look at the derivative with respect 
    # to time can be found by subtracting our frames
    ItGradient = img2 - img1

    # we blur the gradients to get better corner values 
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

        # to avoid errors because of unkown problems with how our corners are being accessed
        if(xCoordinate < height):
            plt.arrow(xCoordinate, yCoordinate, 5 * u[xCoordinate][yCoordinate], 5 * v[xCoordinate][yCoordinate], **kwargsParams)
        else:
            plt.arrow(xCoordinate, yCoordinate, 5 * u[yCoordinate][xCoordinate], 5 * v[yCoordinate][xCoordinate], **kwargsParams)

        # to help further identify the corner images we place circles at the start of the vector
        cv2.circle(img1,(xCoordinate,yCoordinate), 1, 255, -1)
    
    plt.imshow(img1, cmap="gray"), plt.title("Optical Flow on Image 1"), plt.show()


def main():
    img1 = cv2.imread('basketball1.png', 0)
    img2 = cv2.imread('basketball2.png', 0)
    
    LucasKanade(img1, img2)

main()



