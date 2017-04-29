#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dlib
from main.model.tools.boundingbox import BoundingBox
from main.model.config import AVAILABLE_ALGORITHMS
from main.model.algorithm.image_algorithm import ImageAlgorithm


__author__ = 'Iván de Paz Centeno'


class DLibHogSVMFaceDetectionAlgorithm(ImageAlgorithm):
    """
    Algorithm for detection of faces based on HOG + SVM implementation from DLIB.
    """

    def __init__(self, use_gpu=-1):
        """
        Initializes the algorithm.
        :param use_gpu: parameter to set the GPU usage for this algorithm.
        The number represents the index of the GPU in the machine, being -1 the CPU.
        WARNING: This algorithm does not support the usage of GPU yet.
        """

        ImageAlgorithm.__init__(self, DLibHogSVMFaceDetectionAlgorithm.__name__,
                                "DLib Face detection Algorithm based on HOG.")

        self.detector = dlib.get_frontal_face_detector()

    def _process_resource(self, image):
        """
        Processes the specified image in order to get the bounding boxes for the faces.
        :param image: image resource pointing to a valid URI or containing the image content.
                    If the image is not loaded but is pointing to a valid URI, this method
                    will try to load the image from the URI in grayscale.
        :return: an array of bounding boxes
        """

        image_content = self._get_loaded_image_content(image, as_gray=True)

        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        detections = self.detector(image_content, 1)

        metadata_content = []

        for i, d in enumerate(detections):
            bounding_box = BoundingBox(d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
            bounding_box.fit_in_size(image.get_size())
            metadata_content.append(bounding_box)

        return metadata_content


# It needs to be registered here.
AVAILABLE_ALGORITHMS[DLibHogSVMFaceDetectionAlgorithm.__name__] = {
    'prototype': DLibHogSVMFaceDetectionAlgorithm,
    'resource_type': DLibHogSVMFaceDetectionAlgorithm.kind_of_resource(),
    'type': 'DETECTION',
    'subtype': 'FACE',
    'detection_type': BoundingBox
}
