#!/usr/bin/env python
# -*- coding: utf-8 -*-

from main.model.algorithm.algorithm import Algorithm
from main.model.resource.image import Image


__author__ = "Ivan de Paz Centeno"


class ImageAlgorithm(Algorithm):
    """
    Algorithm that process images.
    """

    def is_resource_processable(self, resource):
        """
        Determines if a resource is procesable by this algorithm or not.
        :param resource: Resource to check compaitibility with the algorithm
        :return: True if compatible. False othwerise.
        """
        return isinstance(resource, Image)

    @staticmethod
    def _get_loaded_image_content(image, as_gray=True):
        """
        Retrieves the image content in loaded state.
        :param resource: image resource.
        :param as_gray: flag to set if the image is in grayscale or not.
        :param size_normalizer:
        :return: image content in numpy array format.
        """

        if not image.is_loaded():
            image.load_from_uri(as_gray=as_gray)

        image_content = image.get_blob(True)

        return image_content

    @staticmethod
    def kind_of_resource():
        """
        Returns the kind of resource of this algorithm
        :return:
        """
        return Image
