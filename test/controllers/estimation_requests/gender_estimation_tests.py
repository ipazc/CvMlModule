#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64
import json
import unittest
import flask

from urllib.parse import quote

import main.model.algorithm.estimation.gender.levi_hassner_cnn_gender_estimation_algorithm
import main.services.image.algorithm_service

from main.controllers.controller_factory import ControllerFactory
from main.model.config import Config
from main.model.resource.image import Image

__author__ = 'Ivan de Paz Centeno'


class GenderEstimationControllerTests(unittest.TestCase):
    """
    Tests the class controller AgeEstimationController
    """

    @classmethod
    def setUpClass(cls):
        """
        Sets up the web app in testing mode, with all the modules loaded.
        """
        cls.app = flask.Flask(__name__)
        cls.app.config['TESTING'] = True
        config = Config(ignore_service_when_algorithm_not_available=True)
        cls.factory = ControllerFactory(cls.app, config)

        cls.gender_estimation_controller = cls.factory.gender_estimation_controller()

        cls.gender_estimation_request_url = {
            "uri": "/estimation-requests/gender/face/uri",
            "stream": "/estimation-requests/gender/face/stream",
            "base64": "/estimation-requests/gender/face/base64",
            "services": "/estimation-requests/gender/services"
        }

    @classmethod
    def tearDownClass(cls):
        """
        Resources to release when the class' tests are over.
        """
        # We need to release every service loaded by the factory and not only the services
        # injected to the controllers.
        cls.factory.release_all()

    def get_services(self):
        """
        Retrieves the services for gender estimation.
        :return: Services in JSON format
        """

        with self.app.test_client() as client:
            rv = client.get(self.gender_estimation_request_url["services"])
            response = json.loads(str(rv.data, 'UTF-8'))

        return response

    def send_request(self, content, type, extra_params):
        """
        Auxiliary method to send requests to the REST API.
        This method is not a unit-test as its name does not start with the 'test_' keyword.
        :param content: content of the image to analyze.
        :param type: String setting the kind of request to make (one of the keys of gender_estimation_request_url)
        :param extra_params: dictionary with parameters and values to append to the request.
        :return:
        """

        request_url = self.gender_estimation_request_url[type]

        # We append the extra params to the request url
        if len(extra_params) > 0:
            first_key = list(extra_params.keys()).pop()
            first_value = extra_params[first_key]

            del extra_params[first_key]

            # First is always different because it starts with "?"
            request_url += "?{}={}".format(first_key, quote(first_value))

            # The rest begins with "&"
            for key, value in extra_params.items():
                request_url += "&{}={}".format(key, quote(value))

        with self.app.test_client() as client:
            rv = client.put(request_url, data=content)
            response = json.loads(str(rv.data, 'UTF-8'))

        return response

    def test_detect_stream_service_parameter(self):
        """
        STREAM can work with default and different services.
        :return:
        """

        image = Image("main/samples/example_image.jpg")
        image.load_from_uri(as_gray=True)

        jpeg_content = image.get_jpeg()
        response = self.send_request(jpeg_content, "stream", {
            "service": "caffe-cnn-levi-hassner-gender-estimation"
        })

        response2 = self.send_request(jpeg_content, "stream", {
        })

        response3 = self.send_request(jpeg_content, "stream", {
            "service": "default"
        })

        response4 = self.send_request(jpeg_content, "stream", {
            "service": "INVALID SERVICE TEST"
        })

        self.assertEqual(len(response), 1)
        self.assertEqual(len(response2), 1)
        self.assertEqual(len(response3), 1)
        self.assertEqual(len(response4), 1)
        self.assertIn("Gender", response)
        self.assertIn("Gender", response2)
        self.assertIn("Gender", response3)

        self.assertEqual(set(response), set(response2))
        self.assertEqual(set(response2), set(response3))
        self.assertEqual(response4, {'message': "Service name 'INVALID SERVICE TEST' not valid."})

    def test_detect_stream(self):
        """
        API-Rest URL /estimation-requests/gender/face/stream works for stream requests.
        """

        image = Image("main/samples/example_image.jpg")

        # Grayscale
        image.load_from_uri(as_gray=True)

        jpeg_content = image.get_jpeg()
        response = self.send_request(jpeg_content, "stream", {})
        self.assertEqual(len(response), 1)
        self.assertIn("Gender", response)

        # Color
        image.load_from_uri(as_gray=False)

        jpeg_content = image.get_jpeg()
        response = self.send_request(jpeg_content, "stream", {})
        self.assertEqual(len(response), 1)
        self.assertIn("Gender", response)

        jpeg_content = b'asdasd1io23u897das0dasdoasijdoasidja'
        response = self.send_request(jpeg_content, "stream", {})
        self.assertEqual(response, {'message': "Content of file not valid: Resource was empty. "
                                               "Couldn't perform the analysis on an empty resource."})

    def test_detect_base64(self):
        """
        API-Rest URL /estimation-requests/gender/face/base64 works for base64 requests.
        """

        image = Image("main/samples/image1.jpg")

        # Black and white
        image.load_from_uri(as_gray=True)

        jpeg_b64_content = base64.b64encode(image.get_jpeg())
        response = self.send_request(jpeg_b64_content, "base64", {})
        self.assertEqual(len(response), 1)
        self.assertIn("Gender", response)


        # Color
        image.load_from_uri(as_gray=False)

        jpeg_b64_content = base64.b64encode(image.get_jpeg())
        response = self.send_request(jpeg_b64_content, "base64", {})
        self.assertEqual(len(response), 1)
        self.assertIn("Gender", response)

        jpeg_b64_content = base64.b64encode(b'asdasd1io23u897das0dasdoasijdoasidja')
        response = self.send_request(jpeg_b64_content, "base64", {})
        self.assertEqual(response, {'message': "Content of file not valid: Resource was empty. "
                                               "Couldn't perform the analysis on an empty resource."})

    def test_get_services(self):
        """
        Face detection services are visible in the API-Rest URL /estimation-requests/gender/services
        :return:
        """
        services_list = self.get_services()
        self.assertIn("available_services", services_list)
        self.assertGreater(len(services_list["available_services"]), 0)


if __name__ == '__main__':
    unittest.main()