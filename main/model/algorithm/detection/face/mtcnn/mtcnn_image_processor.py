#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy

__author__ = "Ivan de Paz Centeno"


class MTCNNImageProcessor(object):
    """
    Performs some operations for an image in order to be passed to the CNN.
    """

    def __init__(self, image, minsize, threshold, factor):
        self.image = image
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor

        self.total_boxes = numpy.zeros((0, 9), numpy.float)
        self.points = []

    def get_scales(self, fast_resize):
        """
        Retrieves the scales for the image.
        :param fast_resize: Flag to specify if the image should be fast resized or not.
        :return: list of scaled images.
        """
        (width, height) = (self.image.shape[1], self.image.shape[0])

        min_side = min(height, width)

        float_image = self.image.astype(float)

        minimum_scale = 12.0 / self.minsize
        min_side *= minimum_scale

        # create scale pyramid
        factor_count = 0

        scaled_images = []

        while min_side >= 12:
            scale = minimum_scale * pow(self.factor, factor_count)
            scaled_width = int(numpy.ceil(width * scale))
            scaled_height = int(numpy.ceil(height * scale))

            min_side *= self.factor
            factor_count += 1

            if fast_resize:
                im_data = (float_image - 127.5) * 0.0078125  # [0,255] -> [-1,1]
                im_data = cv2.resize(im_data, (scaled_width, scaled_height))  # default is bilinear
            else:
                im_data = cv2.resize(float_image, (scaled_width, scaled_height))  # default is bilinear
                im_data = (im_data - 127.5) * 0.0078125  # [0,255] -> [-1,1]

            im_data = numpy.swapaxes(im_data, 0, 2)
            im_data = numpy.array([im_data], dtype=numpy.float)

            scaled_images.append([im_data, scale, scaled_width, scaled_height])

        return scaled_images
