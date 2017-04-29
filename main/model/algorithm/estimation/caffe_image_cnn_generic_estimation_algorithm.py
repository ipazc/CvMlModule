#!/usr/bin/env python
# -*- coding: utf-8 -*-
from main.model.algorithm.image_algorithm import ImageAlgorithm

__author__ = "Ivan de Paz Centeno"


class CaffeImageCNNGenericEstimationAlgorithm(ImageAlgorithm):
    """
    Generic algorithm for estimations based on CNN from Caffe for images.
    This is a virtual class and should be inherited.

    """

    def _process_resource(self, image):
        """
        Processes the specified image in order to get the estimation from the CNN.
        :param image: image resource pointing to a valid URI or containing the image content.
                    If the image is not loaded but is pointing to a valid URI, this method
                    will try to load the image from the URI in grayscale.
        :return: the estimation result wrapped in a list
        """

        assert self.estimator, "Estimator for the caffe CNN is not initialized."

        image_content = self._get_loaded_image_content(image, as_gray=True)
        estimated_gender = self.estimator.predict_image(image_content)
        return [estimated_gender]
