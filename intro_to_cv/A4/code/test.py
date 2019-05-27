# CSC320 Winter 2019
# Assignment 4
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
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
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
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
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
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################
    img_row, img_col = source_patches.shape[0], source_patches.shape[1]
    max_iteration = (-(np.log(w) / np.log(alpha))).astype(int)
    w_ai = w * np.power(alpha * np.ones(max_iteration), np.arange(max_iteration)).reshape(-1, 1)

    if odd_iteration:
        start_row, stop_row, start_col, stop_col, step = 0, img_row, 0, img_col, 1
    else:
        start_row, stop_row, start_col, stop_col, step = img_row - 1, -1, img_col - 1, -1, -1

    for row in range(start_row, stop_row, step):
        for col in range(start_col, stop_col, step):
            center = [row, col]
            curr_heap = f_heap[row][col]
            curr_dict = f_coord_dictionary[row][col]
            for k in range(len(f_heap[0][0])):
                if propagation_enabled:
                    if 0 <= row + curr_heap[k][2][0] < img_row and 0 <= col + curr_heap[k][2][1] < img_col:
                        curr_displacement = f_heap[row][col][k][2]
                        curr_target = center + curr_displacement
                        if tuple(curr_displacement) not in curr_dict:
                            error_distance = (-1) * distance(source_patches[row, col],
                                                             target_patches[curr_target[0], curr_target[1]])
                            if error_distance > curr_heap[0][0]:
                                heappushpop(curr_heap, (error_distance, next(_tiebreaker), (curr_heap[k][2])))
                                curr_dict[tuple(center)] = 0
                                curr_dict.pop(tuple(f_heap[row][col][0][2]), None)

                    if 0 <= row - step < img_row and 0 <= row + f_heap[row - step][col][k][2][0] < img_row:
                        curr_displacement = f_heap[row - step][col][k][2]
                        curr_target = center + curr_displacement
                        if tuple(curr_displacement) not in curr_dict:
                            D_1 = -distance(source_patches[row, col],
                                            target_patches[curr_target[0], curr_target[1]])
                            if D_1 > curr_heap[0][0]:
                                heappushpop(curr_heap, (D_1, next(_tiebreaker), curr_displacement))
                                curr_dict[tuple(curr_displacement)] = 0
                                curr_dict.pop(tuple(curr_heap[0][2]), None)

                    if 0 <= col - step < img_col and 0 <= col + f_heap[row][col - step][k][2][1] < img_col:
                        curr_displacement = f_heap[row][col - step][k][2]
                        curr_target = center + curr_displacement
                        if tuple(curr_displacement) not in curr_dict:
                            D_2 = -distance(source_patches[row, col],
                                            target_patches[curr_target[0], curr_target[1]])
                            if D_2 > curr_heap[0][0]:
                                heappushpop(curr_heap, (D_2, next(_tiebreaker), curr_displacement))
                                curr_dict[tuple(curr_displacement)] = 0
                                curr_dict.pop(tuple(curr_heap[0][2]), None)

    # for row in range(start_row, stop_row, step):
    #     for col in range(start_col, stop_col, step):
    #         center = [row, col]
    #         curr_heap = f_heap[row][col]
    #         curr_dict = f_coord_dictionary[row][col]
    #         for k in range(len(f_heap[0][0])):
    #             if propagation_enabled:
    #                 curr_displacement = curr_heap[k][2]
    #                 target_center = center + curr_displacement
    #                 # test current position
    #                 if 0 <= target_center[0] < row and 0 <= target_center[1] < col:
    #                     curr_worst_distance = curr_heap[0][0]
    #                     worst_distance_displacement = curr_heap[0][2]
    #                     update_heap_dict(source_patches, target_patches, center, target_center,
    #                                      curr_displacement, curr_heap, curr_dict,
    #                                      curr_worst_distance, worst_distance_displacement)
    #                 # test KNN of the above/below neighbor
    #                 if 0 < row - step < img_row:
    #                     neighbor = center - np.array([step, 0])
    #                     neighbor_heap = f_heap[neighbor[0]][neighbor[1]]
    #                     curr_displacement = neighbor_heap[k][2]
    #                     target_center = center + curr_displacement
    #                     if 0 <= target_center[0] < row and 0 <= target_center[1] < col:
    #                         curr_worst_distance = curr_heap[0][0]
    #                         worst_distance_displacement = curr_heap[0][2]
    #                         update_heap_dict(source_patches, target_patches, center, target_center,
    #                                          curr_displacement, curr_heap, curr_dict,
    #                                          curr_worst_distance, worst_distance_displacement)
    #                 # test KNN of the left/right neighbor
    #                 if 0 < col - step < img_col:
    #                     neighbor = center - np.array([0, step])
    #                     neighbor_heap = f_heap[neighbor[0]][neighbor[1]]
    #                     curr_displacement = neighbor_heap[k][2]
    #                     target_center = center + curr_displacement
    #                     if 0 <= target_center[0] < row and 0 <= target_center[1] < col:
    #                         curr_worst_distance = f_heap[row][col][0][0]
    #                         worst_distance_displacement = curr_heap[0][2]
    #                         update_heap_dict(source_patches, target_patches, center, target_center,
    #                                          curr_displacement, curr_heap, curr_dict,
    #                                          curr_worst_distance, worst_distance_displacement)
                if random_enabled:
                    R_i = np.random.uniform(-1, 1, (max_iteration, 2))
                    search_radius = np.multiply(w_ai, R_i)
                    displacement = curr_heap[k][2]
                    displacement_array = (displacement + search_radius).astype(int)
                    for curr_displacement in displacement_array:
                        target_center = center + curr_displacement
                        if 0 <= target_center[0] < row and 0 <= target_center[1] < col:
                            curr_worst_distance = curr_heap[0][0]
                            worst_distance_displacement = curr_heap[0][2]
                            update_heap_dict(source_patches, target_patches, center, target_center,
                                             curr_displacement, curr_heap, curr_dict,
                                             curr_worst_distance, worst_distance_displacement)
    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    row, col = source_patches.shape[0], source_patches.shape[1]
    # initialize f_heap and f_coord with exactly
    f_heap = [[0 for _ in range(col)] for _ in range(row)]
    f_coord_dictionary = [[0 for _ in range(col)] for _ in range(row)]

    for i in range(row):
        for j in range(col):
            # initialize the heap and dictionary for the current pixel
            center, curr_heap, curr_dict = [i, j], [], {}
            for k in range(f_k.shape[0]):
                curr_displacement = f_k[k, i, j]
                target_center = center + curr_displacement
                # consider the 2-norm distance as the priority and transform to MAX heap
                error_disctance = (-1) * distance(source_patches[center[0]][center[1]],
                                                  target_patches[target_center[0]][target_center[1]])
                # consider the _tiebreaker in the implementation of the heapq
                heappush(curr_heap, (error_disctance, next(_tiebreaker), curr_displacement))
                curr_dict[tuple(curr_displacement)] = 0
            f_heap[i][j], f_coord_dictionary[i][j] = curr_heap, curr_dict
    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    row, col = len(f_heap), len(f_heap[0])
    K = len(f_heap[0][0])
    # initialize the f_k and d_k
    f_k, D_k = np.zeros((K, row, col, 2)), np.zeros((K, row, col))

    for i in range(row):
        for j in range(col):
            # extract the k heaps, from large to small in order
            # which equivalently, distance from small to large
            curr_heap = nlargest(K, f_heap[i][j])
            # each entry in heap (-distance, counter, displacement)
            for k in range(K):
                f_k[k, i, j] = curr_heap[k][2]
                D_k[k, i, j] = (-1) * curr_heap[k][0]
    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    row, col, channel = target.shape[0], target.shape[1], target.shape[2]
    denoised = np.zeros(target.shape)
    f_k, D_k = NNF_heap_to_NNF_matrix(f_heap)

    # the algorithm formula implemented in the reading material
    exp_array = np.exp(-(D_k / h ** 2))
    norm_z = np.sum(exp_array, axis=0)
    weight_array = exp_array / norm_z

    # get the array of target location
    g_coords = make_coordinates_matrix(target.shape)
    target_array = (g_coords + f_k).reshape((-1, 2))
    images = target[target_array[:, 0].astype(int),
                    target_array[:, 1].astype(int)].reshape((-1, row, col, channel))
    for r in range(row):
        for c in range(col):
            for k in range(len(f_heap[0][0])):
                denoised[r, c] += images[k, r, c] * weight_array[k, r, c]
    #############################################

    return denoised


#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################
# This function return the 2-norm error distance
# between the source_patch centered at center and
# the target_patch centered at center + displacement
def distance(source_patch, target_patch):
    abs_patch_diff = np.abs(source_patch - target_patch)
    # set nan to value 0
    abs_patch_diff[np.isnan(abs_patch_diff)] = 0
    error_distance = np.sum(np.square(abs_patch_diff))
    return error_distance ** .5


# This function updates the input dictionary and heap if necessary
def update_heap_dict(source_patches, target_patches, center, target_center,
                     curr_displacement, curr_heap, curr_dict,
                     curr_worst_distance, worst_distance_displacement):
    if tuple(curr_displacement) not in curr_dict:
        error_distance = (-1) * distance(source_patches[center[0]][center[1]],
                                         target_patches[target_center[0]][target_center[1]])
        if error_distance > curr_worst_distance:
            heappushpop(curr_heap, (error_distance, next(_tiebreaker), curr_displacement))
            curr_dict[tuple(curr_displacement)] = 0
            curr_dict.pop(tuple(worst_distance_displacement), None)
#############################################



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

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    # get the desired mapping coordinates [x,y] in the target image
    mapping_coord = make_coordinates_matrix ( target.shape ) + f
    # Extract the matrix of all x-coordinate and y-coordinate, clip
    x_coord = np.clip ( mapping_coord[:, :, 0], 0, target.shape[0] - 1 ).astype ( int )
    y_coord = np.clip ( mapping_coord[:, :, 1], 0, target.shape[1] - 1 ).astype ( int )
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
