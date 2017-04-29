#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from main.model.algorithm.detection.face.dlib_hog_svm_face_detection_algorithm import DLibHogSVMFaceDetectionAlgorithm
from main.model.resource.image import Image


__author__ = 'Iván de Paz Centeno'


class DLibHogSVMFaceDetectionAlgorithmTest(unittest.TestCase):
    """
    Unitary tests for the DLib Face Detection algorithm.
    """

    def setUp(self):
        """
        Basic set up for the unit tests.
        """
        self.algorithm = DLibHogSVMFaceDetectionAlgorithm()
        self.sampleImageToTest = Image("main/samples/image1.jpg")

    def test_detection(self):
        """
        DLib face detection works correctly.
        """

        image_result, time_spent = self.algorithm.process_resource(self.sampleImageToTest)
        image_metadata = image_result.get_metadata()

        self.assertGreater(time_spent, 0)
        self.assertEqual(len(image_metadata), 3)


if __name__ == '__main__':
    unittest.main()
