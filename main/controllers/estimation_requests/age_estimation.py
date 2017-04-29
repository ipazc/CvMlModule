#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from main.controllers.controller import route
from main.controllers.generic_estimation_controller import GenericEstimationController
from main.controllers.image_controller import ImageController
from main.exceptions.invalid_request import InvalidRequest


__author__ = "Ivan de Paz Centeno"


class AgeEstimationController(GenericEstimationController):
    """
    Controller for /estimation-requests/age/ URL
    """

    def __init__(self, flask_web_app, available_services, config):
        """
        Constructor of the Face Detection controller.
        :param flask_web_app: web app from Flask already initialized.
        :param available_services: list of services filtered to be compatible with this controller.
        :param config: config object containing all the service definitions.
        """
        GenericEstimationController.__init__(self, flask_web_app, available_services, config, "ESTIMATION", "AGE")

        self.exposed_methods += [
            self.estimate_age_of_face_from_content_base64,
            self.estimate_age_of_face_from_content_stream,
            self.get_available_services
        ]

        self._init_exposed_methods()

    @route("/estimation-requests/age/services", methods=['GET'])
    def get_available_services(self):
        """
        Retrieves the services available for age estimation.
        """
        return jsonify(ImageController.get_available_services(self))

    @route("/estimation-requests/age/face/base64", methods=['PUT'])
    def estimate_age_of_face_from_content_base64(self):
        """
        Performs an age estimation on a file containing an image of a face, in the given location.
        It is expected to receive a file encoded in base64 as data.

        The requests accepts the following parameters:
          [OPTIONAL]    service=SERVICE_NAME
          [OPTIONAL]    work_in_gray=true/false             # Works in grayscale or not. (default: true)
          [OPTIONAL]    bounding_box=X,Y,Width,Height       # If set, it will crop the image by this for the estimation.

        :return: The detection result in JSON format (range of age).
        """

        request, service_name, work_in_gray, bounding_box = self._get_common__parameters()

        service = self._get_most_suitable_service(service_name)
        content = self._get_raw_content_validated(is_base64=True)
        image = self._build_image_from_content(content, work_in_gray)

        return self._generic_request(image, bounding_box, service)

    @route("/estimation-requests/age/face/stream", methods=['PUT'])
    def estimate_age_of_face_from_content_stream(self):
        """
        Performs an age estimation on a file containing an image of a face, in the given location.
        It is expected to receive a raw content of an image.

        The requests accepts the following parameters:
          [OPTIONAL]    service=SERVICE_NAME
          [OPTIONAL]    work_in_gray=true/false             # Works in grayscale or not. (default: true)
          [OPTIONAL]    bounding_box=X,Y,Width,Height       # If set, it will crop the image by this for the estimation.

        :return: The detection result in JSON format (range of age).
        """

        request, service_name, work_in_gray, bounding_box = self._get_common__parameters()

        service = self._get_most_suitable_service(service_name)
        content = self._get_raw_content_validated(is_base64=False)
        image = self._build_image_from_content(content, work_in_gray)

        return self._generic_request(image, bounding_box, service)
