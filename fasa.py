#!/usr/bin/env python


# @author: Sardor Abdirayimov 
# @inspired by https://mpatacchiola.github.io/blog/ and
# https://github.com/mpatacchiola/deepgaze/tree/master

import numpy as np
import cv2
import sys
from timeit import default_timer as timer

DEBUG = False

class FasaSaliencyMapping:
    """Implementation of the FASA 
    (Fast, Accurate, and Size-Aware Salient Object Detection) Algorithm
    Paper link: https://link.springer.com/chapter/10.1007/978-3-319-16811-1_34
    """
    def __init__(self, image_h, image_w):
        """Init the classifier"""
        self.image_rows = image_h
        self.image_cols = image_w
        self.salient_image = np.zeros((image_h, image_w), dtype=np.uint8)

        self.mean_vector = np.array([0.555,0.6449, 0.0002, 0.0063]) # where these numbers came
        # What is covariance matrix. Yes you undestand it, but you do not benefit from it!
        self.covariance_matrix_inverse = np.array([[43.3777, 1.7633, -0.4059, 1.0997],
                                                   [1.7633, 40.7221, -0.0165, 0.0447],
                                                   [-0.4059, -0.0165, 87.0455, -3.2744],
                                                   [1.0997, 0.0447, -3.2744, 125.1503]])
    
    def _calculate_histogram(self, image, tot_bins=8):
        """
        Conversion from BGR to LAB color space.
        In addition, the min/max value for each channel is calculated
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # they commmented it

        minL, maxL, _, _ = cv2.minMaxLoc(image[:,:, 0])
        minA, maxA, _, _ = cv2.minMaxLoc(image[:,:, 1])
        minB, maxB, _, _ = cv2.minMaxLoc(image[:,:, 2])

        # Quantization ranges
        self.L_range = np.linspace(minL, maxL, num=tot_bins, endpoint=False)
        self.A_range = np.linspace(minA, maxA, num=tot_bins, endpoint=False)
        self.B_range = np.linspace(minB, maxB, num=tot_bins, endpoint=False)

        # Here the image quantized using the disscrete bins is created.
        self.image_quantized = np.dstack((np.digitize(image[:, :, 0], self.L_range, right=False),
                                          np.digitize(image[:, :, 1], self.A_range, right=False),
                                          np.digitize(image[:, :, 2], self.B_range, right=False)))
        self.image_quantized -= 1 # to fit in range [0,7]

        #it maps the 3D index of hist in a flat 1D array index
        self.map_3d_1d = np.zeros((tot_bins, tot_bins, tot_bins), dtype=np.int32)

        self.histogram = cv2.calcHist([image], channels=[0, 1, 2], mask=None,
                                      histSize=[tot_bins, tot_bins, tot_bins],
                                      ranges=[minL, maxL, minA, maxA, minB, maxB])
        image_indeces = np.vstack((self.image_quantized[:, :, 0].flat,
                                   self.image_quantized[:, :, 1].flat,
                                   self.image_quantized[:, :, 2].flat)
                                   ).astype(np.uint32)
        image_linear = np.ravel_multi_index(image_indeces, (tot_bins, tot_bins, tot_bins))

        # Getting the linear ID index of unique colours
        self.index_matrix = np.transpose(np.nonzero(self.histogram))
        hist_index = np.where(self.histogram > 0)
        unique_color_linear = np.ravel_multi_index(hist_index, (tot_bins, tot_bins, tot_bins))
        self.number_of_colors = np.amax(self.index_matrix.shape)
        self.centx_matrix = np.zeros(self.number_of_colors)
        self.centy_matrix = np.zeros(self.number_of_colors)
        self.centx2_matrix = np.zeros(self.number_of_colors)
        self.centy2_matrix = np.zeros(self.number_of_colors)

        # Using the numpy method where() to find the location of each unique colours
        counter = 0
        for i in unique_color_linear:
            where_y, where_x = np.unravel_index(np.where(image_linear == i), (self.image_rows, self.image_cols))
            #where_x = np.where(image_linear == i)[1]
            #where_y = np.where(image_linear == i)[0]
            self.centx_matrix[counter] = np.sum(where_x)
            self.centy_matrix[counter] = np.sum(where_y)
            self.centx2_matrix[counter] = np.sum(np.power(where_x, 2))
            self.centy2_matrix[counter] = np.sum(np.power(where_y, 2))
            counter += 1
        return image
    
    def _precompute_parameters(self, sigmac=16):
        """
        Semi-Vectorized versionof the precomputer parameters function
        It runs at 0.003 seconds ona squared 400x400 pixel image.
        It returns number of colors and estimate the color_distance_matrix

        @param sigmac: the scalar used in the exponential (default=16)
        @return: the number of unique colors
        """
        L_centroid, A_centroid, B_centroid = np.meshgrid(self.L_range, self.A_range, self. B_range)
        self.unique_pixels = np.zeros((self.number_of_colors, 3))

        if sys.version_info[0] == 2:
            color_range = xrange(0, self.number_of_colors)
        else:
            color_range =  range(0, self.number_of_colors)
        
        for i in color_range:
            i_index = self.index_matrix[i, :]
            L_i = L_centroid[i_index[0], i_index[1], i_index[2]]
            A_i = A_centroid[i_index[0], i_index[1], i_index[2]]
            B_i = B_centroid[i_index[0], i_index[1], i_index[2]]
            self.unique_pixels[i] = np.array([L_i, A_i, B_i])
            self.map_3d_1d[i_index[0], i_index[1], i_index[2]] = i
        
        color_difference_matrix = np.sum(np.power(self.unique_pixels[:, np.newaxis] - self.unique_pixels, 2), axis=2)
        self.color_distance_matrix = np.sqrt(color_difference_matrix)
        self.exponential_color_distance_matrix = np.exp(-np.divide(color_difference_matrix, (2 * sigmac * sigmac))) # TODO: why minus is far away
        return self.number_of_colors
    
    def _bilateral_filtering(self):
        """
        Applying the bilateral filtering to the matrices.
        @return: mx, my, Vx, Vy
        """
        # Obtaining the values through vectorized operations (very efficient)
        self.contrast = np.dot(self.color_distance_matrix, self.histogram[self.histogram > 0])
        normalization_array = np.dot(self.exponential_color_distance_matrix, self.histogram[self.histogram > 0])
        self.mx = np.dot(self.exponential_color_distance_matrix, self.centx_matrix)
        self.my = np.dot(self.exponential_color_distance_matrix, self.centy_matrix)
        mx2 = np.dot(self.exponential_color_distance_matrix, self.centx2_matrix)
        my2 = np.dot(self.exponential_color_distance_matrix, self.centy2_matrix)

        # Normalizing the vectors
        self.mx = np.divide(self.mx, normalization_array)
        self.my = np.divide(self.my, normalization_array)
        mx2 = np.divide(mx2, normalization_array)
        my2 = np.divide(my2, normalization_array)
        self.Vx = np.absolute(np.subtract(mx2, np.power(self.mx, 2))) # TODO: understand why some negative values appear
        self.Vy = np.absolute(np.subtract(my2, np.power(self.my, 2)))
        return self.mx, self.my, self.Vx, self.Vy
    
    def _calculate_probability(self):
        """
        Vectorized version of the probability estimation.
        @return: a vector shape_probability of shape (number_of_colors)
        """
        g = np.array([np.sqrt(12 * self.Vx) / self.image_cols,
                      np.sqrt(12 * self.Vy) / self.image_rows,
                      (self.mx - (self.image_cols / 2.0)) / float(self.image_cols),
                      (self.my - (self.image_rows / 2.0)) / float(self.image_rows)])
        X = (g.T - self.mean_vector)
        Y = X
        A = self.covariance_matrix_inverse
        result = (np.dot(X, A) * Y).sum(1) # This line does the trick
        self.shape_probability = np.exp(- result / 2) # TODO: Why here is minus is far away
        return self.shape_probability
    
    def _compute_saliency_map(self):
        """
        Fast Vectorized version of the saliency map estimation
        @return: the saliency vector
        """
        self.saliency = np.multiply(self.contrast, self.shape_probability)
        a1 = np.dot(self.exponential_color_distance_matrix, self.saliency)
        a2 = np.sum(self.exponential_color_distance_matrix, axis=1)
        self.saliency = np.divide(a1, a2)
        minVal, maxVal, _, _ = cv2.minMaxLoc(self.saliency)
        self.saliency = self.saliency - minVal
        self.saliency = 255 * self.saliency / (maxVal - minVal) + 1e-3
        return self.saliency
    
    def returnMask(self, image, tot_bin=8, format="BGR2LAB"):
        """
        Return the saliency mask of the input image.
        @param: image to process
        @param: tot_bin: number of total bins used in histogram
        @param: format conversation
        @return: the saliency mask
        """
        if format == "BGR2LAB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif format == "BGR2RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif format == "RGB2LAB":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif format == "RGB" or format == "BGR" or format == "LAB":
            pass
        else:
            raise ValueError("Input eror [format error]")
        
        if DEBUG: start = timer()
        self._calculate_histogram(image, tot_bins=tot_bin)
        if DEBUG: end = timer()
        if DEBUG: print("---%s calculate_histogram seconds ---" % (end - start))
        if DEBUG: start = timer()
        number_of_colors = self._precompute_parameters()
        if DEBUG: end = timer()
        if DEBUG: print("--- number of colors: " + str(number_of_colors) + " ---")
        if DEBUG: print("--- %s precompute_parameters seconds ---" % (end - start))
        if DEBUG: start = timer()
        self._bilateral_filtering()
        if DEBUG: end = timer()
        if DEBUG: print("--- %s bilateral_filtering seconds ---" % (end - start))
        if DEBUG: start = timer()
        self._calculate_probability()
        if DEBUG: end = timer()
        if DEBUG: print("--- %s calculate_probability seconds ---" % (end-start))
        if DEBUG: start = timer()
        self._compute_saliency_map()
        if DEBUG: end = timer()
        if DEBUG: print("--- %s compute_saliency_map seconds ---" % (end - start))
        if DEBUG: start = timer()
        it = np.nditer(self.salient_image, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            y = it.multi_index[0]
            x = it.multi_index[1]
            #L_id = self.L_id_matrix[y, x]
            #A_id = self.A_id_matrix[y, x]
            #B_id = sel.f.B_id_matrix[y,x]
            index = self.image_quantized[y, x]
            index = self.map_3d_1d[index[0], index[1], index[2]]
            it[0] = self.saliency[index]
            it.iternext()
        if DEBUG: end = timer()
        #ret, self.salient_image = cv2.threshold(self.salient_image, 150, 255, cv2.THRESH_BINARY)
        if DEBUG: print("--- %s returnMask 'iteration part' seconds ---" % (end - start))
        return self.salient_image

