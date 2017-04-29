#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

from main.exceptions.image_not_loaded import ImageNotLoaded
from main.model.normalizer.normalizer import Normalizer

__author__ = 'Iván de Paz Centeno'


class AbsoluteSizeNormalizer(Normalizer):
    """
    Resizes the image blob to a specific size.
    """

    def __init__(self, width, height, keep_aspect_ratio=False):
        """
        Constructor for the size normalizer.
        :param width: width in pixels to resize the blobs.
        :param height: height in pixels to resize the blobs.
        """
        self.width = int(width)
        self.height = int(height)
        self.keep_aspect_ratio = keep_aspect_ratio

    def apply(self, image):
        """
        Applies the resize normalizer to the specified image.
        :param image: Image to resize. It must be loaded in order for this method to work successfully.
        :return:
        """
        if not image.is_loaded():
            raise ImageNotLoaded("The image \"{}\" is not loaded. It is required to be loaded in order for this "
                                 "normalizer ({}) to work.".format(image.get_uri(), self.__class__.__name__))

        size = [self.width, self.height][::-1]

        if self.keep_aspect_ratio:

            # We find which shape index is bigger than the other
            shape = image.get_blob().shape[:2]

            if shape[0] > size[0]:
                ratio = size[0] / shape[0]
                shape = (size[0], int(shape[1] * ratio))

            if shape[1] > size[1]:
                ratio = size[1] / shape[1]
                shape = (int(shape[0] * ratio), size[1])

            dim = shape

        else:
            dim = (size[0], size[1])

        blob = cv2.resize(image.get_blob(), dim[::-1], interpolation=cv2.INTER_AREA)
        return image.clone(blob_content=blob)

    @classmethod
    def fromstring(cls, size, keep_aspect_ratio=False):
        """
        Creates the instance from a string of the format WIDTHxHEIGHT.
        :param size: string with the format WIDTHxHEIGHT.
        :return: instance of the class
        """

        sizes = size.split('x')

        if len(sizes) != 2:
            raise Exception("Format of size \"{}\" is not valid! It must be WIDTHxHEIGHT. Example: 1024x768".format(size))

        return cls(sizes[0], sizes[1], keep_aspect_ratio=keep_aspect_ratio)
