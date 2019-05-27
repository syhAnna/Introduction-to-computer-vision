## CSC320 Winter 2019 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv

# If you wish to import any additional modules
# or define other utility functions, 
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing 
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing 
    # algorithms. These images are initialized to None and populated/accessed by 
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = { 
            'backA': None, 
            'backB': None, 
            'compA': None, 
            'compB': None, 
            'colOut': None,
            'alphaOut': None, 
            'backIn': None, 
            'colIn': None, 
            'alphaIn': None, 
            'compOut': None, 
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self): 
        return {
            'backA':{'msg':'Image filename for Background A Color','default':None},
            'backB':{'msg':'Image filename for Background B Color','default':None},
            'compA':{'msg':'Image filename for Composite A Color','default':None},
            'compB':{'msg':'Image filename for Composite B Color','default':None},
        }
    # Same as above, but for the output arguments
    def mattingOutput(self): 
        return {
            'colOut':{'msg':'Image filename for Object Color','default':['color.tif']},
            'alphaOut':{'msg':'Image filename for Object Alpha','default':['alpha.tif']}
        }
    def compositingInput(self):
        return {
            'colIn':{'msg':'Image filename for Object Color','default':None},
            'alphaIn':{'msg':'Image filename for Object Alpha','default':None},
            'backIn':{'msg':'Image filename for Background Color','default':None},
        }
    def compositingOutput(self):
        return {
            'compOut':{'msg':'Image filename for Composite Color','default':['comp.tif']},
        }
    
    # Copy the output of the triangulation matting algorithm (i.e., the 
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the 
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################
            
    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        img = cv.imread(fileName)

        if img is None:
            msg = 'Fail read image {}'.format(fileName)
        else:
            success = True
            self._images[key] = img.astype(np.float64)
        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the 
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63. 
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        img = self._images[key]

        if img is None:
            msg = 'Fail write image {}'.format(fileName)
        else:
            img = np.clip(np.array(self._images[key]), 0., 255.)
            success = True
            try:
                cv.imwrite(fileName, img.astype(np.uint8))
            except cv.error:
                print "OpenCV system error"
                exit()
        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # ojbect. 
    def triangulationMatting(self):
        """
success, errorMessage = triangulationMatting(self)
        
        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        compA = self._images['compA']
        compB = self._images['compB']
        backA = self._images['backA']
        backB = self._images['backB']

        if (compA is None) or (compB is None) or (backA is None) or (backB is None):
            msg = 'Exist None input'
        else:
            imshapes = [compA.shape, compB.shape, backA.shape, backB.shape]
            if not all(shape == imshapes[0] for shape in imshapes):
                msg = 'Input image size not identical'
            else:
                compA_b, compB_b, backA_b, backB_b = compA[:, :, 0], compB[:, :, 0], backA[:, :, 0], backB[:, :, 0]
                compA_g, compB_g, backA_g, backB_g = compA[:, :, 1], compB[:, :, 1], backA[:, :, 1], backB[:, :, 1]
                compA_r, compB_r, backA_r, backB_r = compA[:, :, 2], compB[:, :, 2], backA[:, :, 2], backB[:, :, 2]

                # formula derived from the reading paper p263-p264
                alpha_num = (compA_b - compB_b) * (backA_b - backB_b) + \
                            (compA_g - compB_g) * (backA_g - backB_g) + \
                            (compA_r - compB_r) * (backA_r - backB_r)
                alpha_denum = (backA_b - backB_b) ** 2 + (backA_g - backB_g) ** 2 + (backA_r - backB_r) ** 2

                # when one of the pixel's alpha_denum == 0, i.e.: exist similar pixel in the background
                # By the last line in the Figure 1's description on p.262 of the paper:
                # "pixels in the two backings are identical and the technique fails."
                # therefore set the alphaOut of the similar pixel 1 (i.e.: identified as foreground object)
                temp = np.divide(alpha_num, alpha_denum, out=np.zeros_like(alpha_num), where=alpha_denum != 0)
                alphaOut = 1 - temp
                self._images['alphaOut'] = np.clip(alphaOut, 0, 1) * 255.

                colOut_b = (compA_b - (1. - alphaOut) * backA_b + compB_b - (1. - alphaOut) * backB_b) / 2.
                colOut_g = (compA_g - (1. - alphaOut) * backA_g + compB_g - (1. - alphaOut) * backB_g) / 2.
                colOut_r = (compA_r - (1. - alphaOut) * backA_r + compB_r - (1. - alphaOut) * backB_r) / 2.
                colOut = cv.merge((colOut_b, colOut_g, colOut_r))
                self._images['colOut'] = colOut

                success = True
        #########################################

        return success, msg


    def createComposite(self):
        """
success, errorMessage = createComposite(self)
        
        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
"""

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        alphaIn = self._images['alphaIn']
        colIn = self._images['colIn']
        backIn = self._images['backIn']

        if (alphaIn is None) or (colIn is None) or (backIn is None):
            msg = 'Exist None input'
        else:
            imshapes = [alphaIn.shape, colIn.shape, backIn.shape]
            if not all(shape == imshapes[0] for shape in imshapes):
                msg = 'Input image size not identical'
            else:
                success = True
                compOut = (alphaIn / 255.) * colIn + (1 - alphaIn / 255.) * backIn
                self._images['compOut'] = compOut
        #########################################

        return success, msg

