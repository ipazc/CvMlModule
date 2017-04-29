#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from main.model.normalizer.boundingbox.proportion_size_normalizer import ProportionSizeNormalizer
from main.model.tools.boundingbox import BoundingBox

__author__ = "Ivan de Paz Centeno"


class ProportionSizeNormalizerTest(unittest.TestCase):
    """
    Unitary tests for ProportionSizeNormalizer class.
    """

    def test_apply_proportion(self):
        """
        ProportionSizeNormalizer works correctly with proportions for bounding boxes
        """

        bounding_boxes = [
            BoundingBox(100, 100, 500, 500),
            BoundingBox(50, 100, 200, 500)
        ]

        normalizer = ProportionSizeNormalizer(0.5, 0.3)

        for bounding_box in bounding_boxes:
            result = normalizer.apply(bounding_box)

            result_should_be = [int(bounding_box.get_box()[0] * 0.5), int(bounding_box.get_box()[1] * 0.3),
                                int(bounding_box.get_box()[2] * 0.5), int(bounding_box.get_box()[3] * 0.3)]

            self.assertEqual(result.get_box(), result_should_be)


