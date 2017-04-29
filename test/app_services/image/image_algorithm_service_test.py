#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from main.model.algorithm.detection.face.dlib_hog_svm_face_detection_algorithm import DLibHogSVMFaceDetectionAlgorithm
from main.model.resource.image import Image
from main.services.image.algorithm_service import ImageAlgorithmService
from main.services.status import SERVICE_STOPPED

__author__ = 'Iván de Paz Centeno'

import unittest


class ImageAlgorithmServiceTest(unittest.TestCase):
    """
    This will check the whole stack of services.
    """

    def setUp(self):
        """
        Initialization for each testing method. This will set up the service to be
        usable inside each unit test.
        """
        # By default we use the OpenCV face detection algorithm
        self.service = ImageAlgorithmService(DLibHogSVMFaceDetectionAlgorithm, 2, 0)
        self.service.start()

    def tearDown(self):
        """
        When the test is done this closes the service automatically.
        """
        self.service.stop()

    def test_service_start_stop(self):
        """
        Services are able to start and stop all the processes.
        """
        self.service.stop()

        # We assert that it stopped succesfully.
        self.assertEqual(self.service.get_status(), SERVICE_STOPPED)

    def test_service_single_append_request(self):
        """
        Appending a single request works.
        """

        image = Image("main/samples/image1.jpg")
        image.load_from_uri(True)
        result_promise = self.service.append_request(image)

        result = result_promise.get_resource()

        self.assertIsNotNone(result)
        self.assertGreater(len(result.get_metadata()), 0)

    def test_service_multiple_append_request(self):
        """
        Appending multiple request at the same time works.
        """

        # By default we use the OpenCV face detection algorithm
        image = Image("main/samples/image1.jpg")
        image2 = Image("main/samples/kids-6-to-12.jpg")

        image.load_from_uri(True)
        image2.load_from_uri(True)

        result_promise1 = self.service.append_request(image)
        result_promise2 = self.service.append_request(image2)

        result = result_promise1.get_resource()
        result2 = result_promise2.get_resource()

        self.assertIsNotNone(result)
        self.assertIsNotNone(result2)

        self.assertGreater(len(result.get_metadata()), 0)
        self.assertGreater(len(result2.get_metadata()), 0)

    def test_service_multiple_append_request_promises_equal_for_same_resource(self):
        """
        Appending multiple request of a similar resource at the same time
        returns the same promise.
        """

        image1 = Image(uri="main/samples/image1.jpg", image_id="1")
        image2 = Image(uri="main/samples/image1.jpg", image_id="2")

        # Images must be loaded. If not, they will have different md5 hashes
        image1.load_from_uri(True)
        image2.load_from_uri(True)

        result_promise1 = self.service.append_request(image1)
        result_promise2 = self.service.append_request(image2)
        result_promise3 = self.service.append_request(image1)

        # The promise must be equal for both
        self.assertEqual(result_promise1, result_promise3)
        self.assertEqual(result_promise1, result_promise2)

        result1 = result_promise1.get_resource()
        result2 = result_promise2.get_resource()

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)

        self.assertGreater(len(result1.get_metadata()), 0)
        self.assertGreater(len(result2.get_metadata()), 0)

        self.assertEqual(result1, result2)

    #def test_stop_service_before_finishing_promise(self):
    #   """
    #    Service is stoppable while promise hasn't been processed yet.
    #    """

    #    image = Image(uri="main/samples/image1.jpg", image_id="1")
    #    image.load_from_uri(True)

    #    result_promise = self.service.append_request(image)

    #    self.service.stop(False)

    #    # It should block until the resource is processed, and then the service is closed.
    #    result = result_promise.get_resource()

    #    self.assertIsNotNone(result)
    #    self.assertGreater(len(result.get_metadata()), 0)


if __name__ == '__main__':
    unittest.main()
