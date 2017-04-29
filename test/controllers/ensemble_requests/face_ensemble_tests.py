#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64
import json
import unittest
import flask

from urllib.parse import quote

import main.model.algorithm.detection.face.opencv_haar_cascade_face_detection_algorithm
import main.model.algorithm.detection.face.dlib_hog_svm_face_detection_algorithm

import main.model.algorithm.estimation.age.levi_hassner_cnn_age_estimation_algorithm
import main.model.algorithm.estimation.gender.levi_hassner_cnn_gender_estimation_algorithm

import main.services.image.algorithm_service

from main.controllers.controller_factory import ControllerFactory
from main.model.config import Config
from main.model.resource.image import Image

__author__ = 'Ivan de Paz Centeno'


class FaceEnsembleControllerTests(unittest.TestCase):
    """
    Tests the class controller FaceDetectionController
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

        cls.face_ensemble_controller = cls.factory.face_ensemble_controller()

        cls.face_ensemble_request_url = {
            "stream": "/ensemble-requests/faces/detection-estimation-age-gender/stream",
            "base64": "/ensemble-requests/faces/detection-estimation-age-gender/base64",
            "services": "/ensemble-requests/faces/services"
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
        Retrieves the services for face detection.
        :return: Services in JSON format
        """

        with self.app.test_client() as client:
            rv = client.get(self.face_ensemble_request_url["services"])
            response = json.loads(str(rv.data, 'UTF-8'))

        return response

    def send_request(self, content, type, extra_params):
        """
        Auxiliary method to send requests to the REST API.
        This method is not a unit-test as its name does not start with the 'test_' keyword.
        :param content: content of the image to analyze.
        :param type: String setting the kind of request to make (one of the keys of face_detection_request_url)
        :param extra_params: dictionary with parameters and values to append to the request.
        :return:
        """

        request_url = self.face_ensemble_request_url[type]

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

    def test_detect_stream_without_age(self):
        """
        Face-ensemble API-Rest works when requested without age.
        """

        image = Image("main/samples/image1.jpg")
        image.load_from_uri(as_gray=True)

        jpeg_content = image.get_jpeg()
        response = self.send_request(jpeg_content, "stream", {
            "service_age": "none"
        })

        self.assertEqual(len(response), 3)
        ids = []

        for face_index, content in response.items():
            self.assertIn('Bounding_box', content)
            self.assertIn('Gender', content)
            self.assertNotIn('Age_range', content)
            self.assertIn('Face_ID', content)
            self.assertNotIn(face_index, ids)
            ids.append(int(face_index))

    def test_detect_stream_without_gender(self):
        """
        Face-ensemble API-Rest works when requested without gender.
        """

        image = Image("main/samples/image1.jpg")
        image.load_from_uri(as_gray=True)

        jpeg_content = image.get_jpeg()
        response = self.send_request(jpeg_content, "stream", {
            "service_gender": "none"
        })

        self.assertEqual(len(response), 3)
        ids = []

        for face_index, content in response.items():
            self.assertIn('Bounding_box', content)
            self.assertNotIn('Gender', content)
            self.assertIn('Age_range', content)
            self.assertIn('Face_ID', content)
            self.assertNotIn(face_index, ids)
            ids.append(int(face_index))

    def test_detect_stream_without_age_gender(self):
        """
        Face-ensemble API-Rest works when requested without age and gender.
        """

        image = Image("main/samples/image1.jpg")
        image.load_from_uri(as_gray=True)

        jpeg_content = image.get_jpeg()
        response = self.send_request(jpeg_content, "stream", {
            "service_age": "none",
            "service_gender": "none"
        })

        self.assertEqual(len(response), 3)
        ids = []

        for face_index, content in response.items():
            self.assertIn('Bounding_box', content)
            self.assertNotIn('Gender', content)
            self.assertNotIn('Age_range', content)
            self.assertIn('Face_ID', content)
            self.assertNotIn(face_index, ids)
            ids.append(int(face_index))

    def test_detect_stream(self):
        """
        API-Rest URL /ensemble-requests/faces/detection-estimation-age-gender/stream works for stream requests.
        """

        image = Image("main/samples/image1.jpg")
        image.load_from_uri(as_gray=True)

        jpeg_content = image.get_jpeg()
        response = self.send_request(jpeg_content, "stream", {})

        self.assertEqual(len(response), 3)
        ids = []

        for face_index, content in response.items():
            self.assertIn('Bounding_box', content)
            self.assertIn('Gender', content)
            self.assertIn('Age_range', content)
            self.assertIn('Face_ID', content)
            self.assertNotIn(face_index, ids)
            ids.append(int(face_index))

    def test_detect_base64(self):
        """
        API-Rest URL /ensemble-requests/faces/detection-estimation-age-gender/base64 works for base64 requests.
        """

        image = Image("main/samples/image1.jpg")
        image.load_from_uri(as_gray=True)

        jpeg_b64_content = base64.b64encode(image.get_jpeg())
        response = self.send_request(jpeg_b64_content, "base64", {})

        self.assertEqual(len(response), 3)
        ids = []

        for face_index, content in response.items():
            self.assertIn('Bounding_box', content)
            self.assertIn('Gender', content)
            self.assertIn('Age_range', content)
            self.assertIn('Face_ID', content)
            self.assertNotIn(face_index, ids)
            ids.append(int(face_index))

    def test_get_services(self):
        """
        Ensemble services are visible in the API-Rest URL /ensemble-requests/faces/services
        :return:
        """

        ensemble_controllers = ["age_estimation_controller",
                                "gender_estimation_controller",
                                "face_detection_controller"]

        processed_controllers = []

        ensemble_services_dict = self.get_services()

        for controller_name in ensemble_controllers:

            try:
                services = ensemble_services_dict[controller_name]
                processed_controllers.append(controller_name)
                self.assertIn("available_services", services)
                self.assertGreater(len(services), 0)

            except KeyError as exception:
                print("Exception: ", exception.__str__())
                pass

        self.assertEqual(len(set(ensemble_controllers)), len(set(processed_controllers)))


if __name__ == '__main__':
    unittest.main()
