## CSC320 Winter 2019 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    # zero out the unknown pixel
    valid_image = np.divide(np.multiply(filledImage, confidenceImage), 255)

    # numerator: sum of the C(q) of all filled pixels in the patch
    # dealing with the case: the patch somehow out of the boundary of the image
    window, valid = copyutils.getWindow(valid_image, psiHatP._coords, psiHatP._w)
    numerator = np.sum(window * valid)

    # denominator: the size of the patch
    # valid patch size, i.e.: not out of the boundary of the image
    denominator = np.sum(valid)

    # confidences are of type uint8, taking values between 0 and 255.
    C = np.clip((numerator / denominator).astype(np.uint8), 0, 255)
    #########################################

    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#

def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    # OpenCV function converts color image to grayscale image
    gray = cv.cvtColor(inpaintedImage, cv.COLOR_BGR2GRAY)

    # OpenCV function computes horizontal and vertical component of the gradient
    # 3x3 Scharr filter is used which gives better results than 3x3 Sobel filter
    Dx = cv.Scharr(gray, cv.CV_64F, 1, 0)
    Dy = cv.Scharr(gray, cv.CV_64F, 0, 1)

    # zero out the unknown pixel
    Dx_valid = np.divide(np.multiply(Dx, filledImage), 255.)
    Dy_valid = np.divide(np.multiply(Dy, filledImage), 255.)

    # Dealing the case, the patch somehow out of the boundary of the image
    windowDx, validDx = copyutils.getWindow(Dx_valid, psiHatP._coords, psiHatP._w)
    windowDy, validDy = copyutils.getWindow(Dy_valid, psiHatP._coords, psiHatP._w)

    # focus on the pixels inside the patch
    Dx_patch = np.array(windowDx * validDx)
    Dy_patch = np.array(windowDy * validDy)

    # combine Dx and Dy to the total gradient
    D_patch = np.power(Dx_patch, 2) + np.power(Dy_patch, 2)

    Dx = Dx_patch.item(np.argmax(D_patch))
    Dy = Dy_patch.item(np.argmax(D_patch))
    #########################################

    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    window, valid = copyutils.getWindow(fillFront, psiHatP._coords, psiHatP._w)
    valid_window = np.array(window * valid)

    if np.count_nonzero(valid_window) == 1:
        Nx = None
        Ny = None
    else:
        # get the gradient kernel w.r.t a and y direction
        sobel3x = np.array(cv.getDerivKernels(1, 0, 3))
        sobel3y = np.array(cv.getDerivKernels(0, 1, 3))

        # use convolution to get the gradient on x and y direction
        xGradient = np.array(cv.filter2D(valid_window, cv.CV_64F, sobel3x))
        yGradient = np.array(cv.filter2D(valid_window, cv.CV_64F, sobel3y))

        # only focus on the center pixel
        center_index = (window.size - 1) / 2
        dx = xGradient.item(center_index)
        dy = yGradient.item(center_index)
        norm = (dx ** 2 + dy ** 2) ** (1 / 2)

        Nx = dx / norm
        Ny = dy / norm
    #########################################

    return Ny, Nx