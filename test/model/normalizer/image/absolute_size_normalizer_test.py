#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from main.exceptions.image_not_loaded import ImageNotLoaded
from main.model.normalizer.image.absolute_size_normalizer import AbsoluteSizeNormalizer
from main.model.resource.image import Image

__author__ = "Ivan de Paz Centeno"


class AbsoluteSizeNormalizerTest(unittest.TestCase):
    """
    Unitary tests for AbsoluteSizeNormalizer class.
    """

    def setUp(self):
        """
        Basic set up for the unit tests.
        """
        self.image_sample_uri = "main/samples/image1.jpg"

    def test_apply_without_load_raise_exception(self):
        """
        AbsoluteSizeNormalizer raises exception if image is not loaded when applied.
        """

        sample_image_to_test = Image(uri=self.image_sample_uri)
        normalizer = AbsoluteSizeNormalizer(500, 500, keep_aspect_ratio=True)

        with self.assertRaises(ImageNotLoaded):
            normalizer.apply(sample_image_to_test)


    def test_apply_equal_width_height_keeping_aspect_ratio(self):
        """
        AbsoluteSizeNormalizer is able to apply to an image keeping the aspect ratio if the width and height are equal.
        :return:
        """
        size = (500, 500)

        sample_image_to_test = Image(uri=self.image_sample_uri)
        sample_image_to_test.load_from_uri()

        normalizer = AbsoluteSizeNormalizer(size[0], size[1], keep_aspect_ratio=True)
        image_result = normalizer.apply(sample_image_to_test)

        self.assertNotEqual(image_result, sample_image_to_test)
        self.assertNotEqual(image_result.get_blob(), sample_image_to_test.get_blob())

        original_size = sample_image_to_test.get_size()
        new_size = image_result.get_size()

        index, value = max(enumerate(original_size), key=lambda v: v[1])

        self.assertEqual(size[index], new_size[index])
        self.assertLess(new_size[index - 1], size[index - 1])

    def test_apply_different_width_height_keeping_aspect_ratio(self):
        """
        AbsoluteSizeNormalizer is able to apply to an image keeping the aspect ratio if the width and height are
        different.
        :return:
        """

        sample_image_to_test = Image(uri=self.image_sample_uri)
        sample_image_to_test.load_from_uri()

        sizes = [
            (400, 100),
            (100, 400)
        ]

        for size in sizes:
            normalizer = AbsoluteSizeNormalizer(size[0], size[1], keep_aspect_ratio=True)

            image_result = normalizer.apply(sample_image_to_test)

            self.assertNotEqual(image_result, sample_image_to_test)
            self.assertNotEqual(image_result.get_blob(), sample_image_to_test.get_blob())

            new_size = image_result.get_size()

            index, value = min(enumerate(size), key=lambda v: v[1])

            self.assertEqual(size[index], new_size[index])
            self.assertLess(new_size[index - 1], size[index - 1])

    def test_apply_different_width_height_forcefully(self):
        """
        AbsoluteSizeNormalizer is able to apply to an image forcefully a different width and height.
        :return:
        """

        sample_image_to_test = Image(uri=self.image_sample_uri)
        sample_image_to_test.load_from_uri()

        sizes = [
            (400, 100),
            (500, 500),
            (100, 400)
        ]

        for size in sizes:
            normalizer = AbsoluteSizeNormalizer(size[0], size[1], keep_aspect_ratio=False)

            image_result = normalizer.apply(sample_image_to_test)

            self.assertNotEqual(image_result, sample_image_to_test)
            self.assertNotEqual(image_result.get_blob(), sample_image_to_test.get_blob())

            new_size = image_result.get_size()

            self.assertEqual(size, new_size)
