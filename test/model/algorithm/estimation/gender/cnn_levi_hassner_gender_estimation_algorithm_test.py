#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from main.model.algorithm.estimation.gender.levi_hassner_cnn_gender_estimation_algorithm \
    import LeviHassnerCNNGenderEstimationAlgorithm
from main.model.tools.gender import Gender, GENDER_FEMALE
from main.model.resource.image import Image


__author__ = 'Iván de Paz Centeno'


class CNNLeviHassnerGenderEstimationAlgorithmTest(unittest.TestCase):
    """
    Unitary tests for the CNN Gender Estimation algorithm from Levi and Hassner implementation.
    """

    def setUp(self):
        """
        Basic set up for the unit tests.
        """
        self.algorithm = LeviHassnerCNNGenderEstimationAlgorithm()
        self.sampleImageToTest = Image("main/samples/example_image.jpg")

        self.gender_to_match = Gender(GENDER_FEMALE)

    def test_estimation(self):
        """
        CNN (Caffe) Levi & Hassner gender estimation works correctly.
        """

        image_result, time_spent = self.algorithm.process_resource(self.sampleImageToTest)
        image_metadata = image_result.get_metadata()

        predicted_gender = image_metadata[0]

        self.assertGreater(time_spent, 0)
        self.assertEqual(predicted_gender.get_gender(), self.gender_to_match.get_gender())


if __name__ == '__main__':
    unittest.main()
