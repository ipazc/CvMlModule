#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from main.controllers.controller import route
from main.controllers.image_controller import ImageController
from main.exceptions.invalid_request import InvalidRequest

__author__ = "Ivan de Paz Centeno"


class FaceDetectionController(ImageController):
    """
    Controller for /detection-requests/faces/ URL
    """

    def __init__(self, flask_web_app, available_services, config):
        """
        Constructor of the Face Detection controller.
        :param flask_web_app: web app from Flask already initialized.
        :param available_services: list of services filtered to be compatible with this controller.
        :param config: config object containing all the service definitions.
        """
        ImageController.__init__(self, flask_web_app, available_services, config, "DETECTION", "FACE")

        self.exposed_methods += [
            self.detect_face_from_content_base64,
            self.detect_face_from_content_stream,
            self.get_available_services
        ]

        self._init_exposed_methods()

    @route("/detection-requests/faces/services", methods=['GET'])
    def get_available_services(self):
        """
        Retrieves the services available for face detection.
        """
        return jsonify(ImageController.get_available_services(self))

    @route("/detection-requests/faces/base64", methods=['PUT'])
    def detect_face_from_content_base64(self):
        """
        Performs a detection of the given image content in Base64 format.
        It is expected to receive a file encoded in base64 as a data.

        The requests accepts the following parameters:
          [OPTIONAL]    service=SERVICE_NAME
          [OPTIONAL]    work_in_gray=true/false             # Works in grayscale or not. (default: true)

        :return: The detection result in JSON format (bounding boxes).
        """

        request = self._get_validated_request()
        service_name = request.get('service', 'default')
        work_in_gray = request.get('work_in_gray', "true") == "true"

        service = self._get_most_suitable_service(service_name)
        content = self._get_raw_content_validated(is_base64=True)

        image = self._build_image_from_content(content, work_in_gray)

        # This will block the request until the resource is ready.
        result = service.append_request(image).get_resource()

        bounding_boxes = {"bounding_boxes": [bbox.__str__() for bbox in self._retrieve_result_metadata(result)]}

        return jsonify(bounding_boxes)

    @route("/detection-requests/faces/stream", methods=['PUT'])
    def detect_face_from_content_stream(self):
        """
        Performs a detection of the given image content from a stream of data.
        It is expected to receive a raw content of an image.

        The requests accepts the following parameters:
          [OPTIONAL]    service=SERVICE_NAME
          [OPTIONAL]    work_in_gray=true/false             # Works in grayscale or not. (default: true)

        :return: The detection result in JSON format (bounding boxes).
        """

        request = self._get_validated_request()
        service_name = request.get('service', 'default')
        work_in_gray = request.get('work_in_gray', "true") == "true"

        service = self._get_most_suitable_service(service_name)
        content = self._get_raw_content_validated(is_base64=False)

        image = self._build_image_from_content(content, work_in_gray)

        # This will block the request until the resource is ready.
        result = service.append_request(image).get_resource()

        bounding_boxes = {"bounding_boxes": [bbox.__str__() for bbox in self._retrieve_result_metadata(result)]}

        return jsonify(bounding_boxes)
