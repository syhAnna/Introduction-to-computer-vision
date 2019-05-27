# CSC320 Winter 2019
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # initialize best_D, initially each color channel set the maximum value
    best_D = np.ones((source_patches.shape[0], source_patches.shape[1])) * 255 * 3
    max_iteration = (-(np.log(w) / np.log(alpha))).astype(int)
    w_ai = w * np.power(alpha * np.ones(max_iteration), np.arange(max_iteration)).reshape(-1, 1)

    if odd_iteration:
        # loop over each pixel / patch center in source_patches
        for row in range(source_patches.shape[0]):
            for col in range(source_patches.shape[1]):
                center = [row, col]
                curr_offset, curr_distance = new_f[row, col], best_D[row, col]

                # cases for propagation_enabled
                if row == 0 and col != 0:
                    # top-most row, only consider left neighbor
                    candidate_offset = clip_offset(center, new_f[0, col - 1], target_patches)
                    candidate_array = np.array([curr_offset, candidate_offset])
                elif row != 0 and col == 0:
                    # left-most col, only consider up neighbor
                    candidate_offset = clip_offset(center, new_f[row - 1, 0], target_patches)
                    candidate_array = np.array([curr_offset, candidate_offset])
                else:
                    # consider both left and up, top left corner do nothing
                    candidate_offset1 = clip_offset(center, new_f[row - 1, col], target_patches)
                    candidate_offset2 = clip_offset(center, new_f[row, col - 1], target_patches)
                    candidate_array = np.array([curr_offset, candidate_offset1, candidate_offset2])

                if propagation_enabled:
                    curr_offset, error_distance = best_offset(candidate_array, center,
                                                              source_patches, target_patches)
                if random_enabled:
                    R_i = np.random.uniform(-1, 1, (max_iteration, 2))
                    search_radius = np.multiply(w_ai, R_i)
                    offset_array = curr_offset + search_radius
                    target_array = center + offset_array
                    # clip the coordinate in offset_array inside the target_patches
                    target_x = np.clip(target_array[:, 0], 0, target_patches.shape[0] - 1).astype(int)
                    target_y = np.clip(target_array[:, 1], 0, target_patches.shape[1] - 1).astype(int)
                    new_target = np.stack((target_x, target_y), -1)
                    new_offset = new_target - center
                    candidate_array = np.append(new_offset, [curr_offset], axis=0)
                    curr_offset, curr_distance = best_offset(candidate_array, center,
                                                             source_patches, target_patches)

                new_f[center[0], center[1]] = curr_offset
                best_D[center[0], center[1]] = curr_distance
    else:
        # even_iteration, examining offsets in reverse scan order
        source_patches, target_patches = flip_udlr(source_patches), flip_udlr(target_patches)
        # f stored vector, multiply -1 to reverse the direction of the vector
        new_f, best_D = (-1) * flip_udlr(new_f), flip_udlr(best_D)
        new_f, best_D, global_vars = propagation_and_random_search(
            source_patches, target_patches, new_f, alpha, w,
            propagation_enabled, random_enabled, True, best_D, global_vars)
        # flip the result back
        new_f, best_D = (-1) * flip_udlr(new_f), flip_udlr(best_D)
    #############################################

    return new_f, best_D, global_vars

# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # get the desired mapping coordinates [x,y] in the target image
    mapping_coord = make_coordinates_matrix(target.shape) + f
    # Extract the matrix of all x-coordinate and y-coordinate, clip
    x_coord = np.clip(mapping_coord[:, :, 0], 0, target.shape[0] - 1).astype(int)
    y_coord = np.clip(mapping_coord[:, :, 1], 0, target.shape[1] - 1).astype(int)
    # Set [pixel (x,y) of the source] the intensity of [pixel (x,y)+f(x,y) of the target].
    rec_source = target[x_coord, y_coord]
    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))


#######################################################
###  PLACE THE HELPER FUNCTION BETWEEN THESE LINES  ###
#######################################################
# This function flip the input matrix in both
# up/down direction and left/right direction
def flip_udlr(matrix):
    if matrix is not None:
        matrix = np.flipud(matrix)
        matrix = np.fliplr(matrix)
    return matrix


# This function clip the target coordinates
# inside the input image shape, where the
# target coordinate = center coordinate +
# offset coordinate
def clip_offset(center, offset, img):
    candidate = np.add(center, offset)
    x_coord = np.clip(candidate[0], 0, img.shape[0] - 1).astype(int)
    y_coord = np.clip(candidate[1], 0, img.shape[1] - 1).astype(int)
    return np.subtract([x_coord, y_coord], center)


# This function return the vector with the lowest
# D-value in the candidate_array
def best_offset(candidate_array, center, source_patches, target_patches):
    # center coordinate add to each offset in the array
    target_positions = candidate_array + center
    abs_patch_diff = np.abs(source_patches[center[0], center[1]] -
                            target_patches[target_positions[:, 0], target_positions[:, 1]])
    # set nan to maximum 1-norm value 255
    abs_patch_diff[np.isnan(abs_patch_diff)] = 255
    error_distance = np.sum(np.sum(abs_patch_diff, axis=1), axis=1)
    min_index = np.argmin(error_distance)
    return candidate_array[min_index], error_distance[min_index]
#######################################################
