#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from main.model.tools.age_range import AgeRange
from main.model.algorithm.estimation.age.levi_hassner_cnn_age_estimation_algorithm \
    import LeviHassnerCNNAgeEstimationAlgorithm
from main.model.resource.image import Image


__author__ = 'Iván de Paz Centeno'


class CNNLeviHassnerAgeEstimationAlgorithmTest(unittest.TestCase):
    """
    Unitary tests for the CNN Age Estimation algorithm from Levi and Hassner implementation.
    """

    def setUp(self):
        """
        Basic set up for the unit tests.
        """
        self.algorithm = LeviHassnerCNNAgeEstimationAlgorithm()
        self.sampleImageToTest = Image("main/samples/example_image.jpg")

        self.age_to_match = AgeRange(0, 2)

    def test_estimation(self):
        """
        CNN (Caffe) Levi & Hassner age estimation works correctly.
        """

        image_result, time_spent = self.algorithm.process_resource(self.sampleImageToTest)
        image_metadata = image_result.get_metadata()

        predicted_age = image_metadata[0]

        self.assertGreater(time_spent, 0)
        self.assertEqual(predicted_age.get_range(), self.age_to_match.get_range())


if __name__ == '__main__':
    unittest.main()
