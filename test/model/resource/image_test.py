#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import unittest
from main.model.tools.boundingbox import BoundingBox
from main.model.resource.image import Image


__author__ = 'Iván de Paz Centeno'


class ImageTest(unittest.TestCase):
    """
    Unit tests for the Image class.
    """

    def test_load_save(self):
        """
        The image is able to load from a file and save to a file.
        """
        image = Image(uri='main/samples/image1.jpg')
        image.load_from_uri()
        self.assertTrue(image.is_loaded())

        destination = Image(uri='main/samples/output.jpg', blob_content=image.get_blob())
        destination.save_to_uri()
        self.assertTrue(destination.exists())

    def test_crop_image(self):
        """
        Image can be cropped by bounding box.
        """
        image = Image(uri='main/samples/image1.jpg')
        image.load_from_uri()

        boundingbox_to_crop = BoundingBox(940, 219, 186, 185)

        cropped = image.crop_image(boundingbox_to_crop, 'main/samples/result_cropped.jpg')

        self.assertTrue(cropped.is_loaded())
        self.assertLess(cropped.get_size(), image.get_size())

    def test_load_image_from_route_with_spaces(self):
        """
        Image can load from a route with special characters like spaces.
        """
        spaces_route = "main/samples/test_routes/spaces/route with spaces/test space.jpg"
        image = Image(uri=spaces_route)

        self.assertTrue(image.exists())

        image.load_from_uri()
        self.assertTrue(image.is_loaded())

    def test_image_get_size(self):
        """
        Image's get_size methods works successfully returning width and height.
        """
        image = Image(uri='main/samples/image1.jpg')
        self.assertEqual(image.get_size(), ())

        image.load_from_uri()
        size = image.get_size()

        self.assertEqual(len(size), 2)
        self.assertGreaterEqual(size[0], 0)
        self.assertGreaterEqual(size[1], 0)

    def test_image_md5_hash(self):
        """
        Image's md5 represents the image.
        """
        image = Image(uri='main/samples/image1.jpg', image_id="1")
        hash_unloaded = image.md5hash()

        image2 = Image(uri='main/samples/image1.jpg', image_id="2")

        # Without content it is based on ID
        self.assertNotEqual(hash_unloaded, image2.md5hash())

        image.load_from_uri()

        self.assertNotEqual(hash_unloaded, image.md5hash())

        image2.load_from_uri()

        # With content it is based on content
        self.assertEqual(image.md5hash(), image2.md5hash())

    def test_image_str(self):
        """
        The string representation of image behaves as expected.
        """
        image = Image(uri='main/samples/image1.jpg')

        self.assertEqual(image.__str__(), "Image, Loaded: {}, size: {}".format(False, ()))
        image.load_from_uri()

        self.assertEqual(image.__str__(), "Image, Loaded: {}, size: {}".format(True, image.get_size()))

    def test_jpeg(self):
        """
        Image can be converted to JPEG successfully.
        """
        image = Image(uri='main/samples/image1.jpg')

        self.assertEqual(image.get_jpeg(), 0)

        image.load_from_uri()

        self.assertGreater(len(image.get_jpeg()), 0)

    def test_is_gray(self):
        """
        Image's grey check works correctly.
        """
        image = Image(uri='main/samples/image1.jpg')
        image.load_from_uri(True)

        self.assertTrue(image.is_gray())

        image.load_from_uri(False)
        self.assertFalse(image.is_gray())

    def test_image_uint_conversion(self):
        """
        Image can be converted to UInt8.
        """
        image = Image(uri='main/samples/image1.jpg')
        image.load_from_uri()
        image.convert_to_boolean()
        self.assertNotEqual(image.get_blob().dtype, numpy.uint8)
        image.convert_to_uint()
        self.assertEqual(image.get_blob().dtype, numpy.uint8)

    def test_image_boolean_conversion(self):
        """
        Image can be converted to boolean.
        """
        image = Image(uri='main/samples/image1.jpg')
        image.load_from_uri()
        self.assertFalse(image.is_boolean())
        image.convert_to_boolean()
        self.assertTrue(image.is_boolean())

    def test_image_clone(self):
        """
        Image can be cloned.
        """
        image = Image(uri='main/samples/image1.jpg')
        cloned_image = image.clone()

        self.assertNotEqual(image, cloned_image)
        self.assertEqual(image.get_uri(), cloned_image.get_uri())
        self.assertEqual(image.get_id(), cloned_image.get_id())
        self.assertFalse(image.is_loaded())
        self.assertFalse(cloned_image.is_loaded())

        image.load_from_uri()
        cloned_image = image.clone()

        self.assertNotEqual(image, cloned_image)
        self.assertTrue(image.is_loaded())
        self.assertTrue(cloned_image.is_loaded())

        cloned_image = image.clone(uri="/this/is/a/test")
        self.assertNotEqual(image.get_uri(), cloned_image.get_uri())

        cloned_image = image.clone(image_id="test")
        self.assertNotEqual(image.get_id(), cloned_image.get_id())

        cloned_image = image.clone(metadata=["test"])
        self.assertNotEqual(image.get_metadata(), cloned_image.get_metadata())

        cloned_image = image.clone(blob_content=[])
        self.assertNotEqual(image.get_blob(), cloned_image.get_blob())

if __name__ == '__main__':
    unittest.main()
