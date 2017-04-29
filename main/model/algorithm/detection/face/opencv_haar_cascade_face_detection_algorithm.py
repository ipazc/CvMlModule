#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from main.model.algorithm.image_algorithm import ImageAlgorithm
from main.model.tools.boundingbox import BoundingBox
from main.model.config import AVAILABLE_ALGORITHMS


__author__ = 'Iván de Paz Centeno'

CASCADE_DIRECTORY = 'main/data/opencv/haarcascades/haarcascade_frontalface_default.xml'


class OpenCVHaarCascadeFaceDetectionAlgorithm(ImageAlgorithm):
    """
    Algorithm for detection of faces based on Viola&Jones HaarCascades implementation from OpenCV.
    """

    def __init__(self, use_gpu=-1):
        """
        Initializes the algorithm.
        :param use_gpu: parameter to set the GPU usage for this algorithm.
        The number represents the index of the GPU in the machine, being -1 the CPU.
        WARNING: This algorithm does not support the usage of GPU yet.
        """

        ImageAlgorithm.__init__(self, OpenCVHaarCascadeFaceDetectionAlgorithm.__name__,
                                "OpenCV Face detection Algorithm based on Haar cascade (Viola&Jones)")

        # Depending on the interpreter working directory, there could be different possibilities
        self.detector = cv2.CascadeClassifier(CASCADE_DIRECTORY)

    def _process_resource(self, image):
        """
        Processes the specified image in order to get the bounding boxes for the faces.
        :param image: image resource pointing to a valid URI or containing the image content.
                    If the image is not loaded but is pointing to a valid URI, this method
                    will try to load the image from the URI in grayscale.
        :return: an array of bounding boxes.
        """

        image_content = self._get_loaded_image_content(image, as_gray=True)

        detections = self.detector.detectMultiScale(image_content, 1.3, 5)

        metadata_content = []

        for (x, y, width, height) in detections:
            bounding_box = BoundingBox(int(x), int(y), int(width), int(height))
            bounding_box.fit_in_size(image.get_size())
            metadata_content.append(bounding_box)

        return metadata_content


# It needs to be registered here.
AVAILABLE_ALGORITHMS[OpenCVHaarCascadeFaceDetectionAlgorithm.__name__] = {
    'prototype': OpenCVHaarCascadeFaceDetectionAlgorithm,
    'resource_type': OpenCVHaarCascadeFaceDetectionAlgorithm.kind_of_resource(),
    'type': 'DETECTION',
    'subtype': 'FACE',
    'detection_type': BoundingBox
}
