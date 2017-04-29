#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from main.controllers.image_controller import ImageController
from main.exceptions.invalid_request import InvalidRequest
from main.model.tools.boundingbox import BoundingBox


__author__ = "Ivan de Paz Centeno"


class GenericEstimationController(ImageController):
    """
    Virtual controller for /estimation-requests/ URL.
    Override this controller to handle estimation requests.
    """

    @staticmethod
    def _get_common__parameters(should_have_uri=False):
        """
        Retrieves the common parameters for age estimation requests.
        :param should_have_uri: if the flag is True, the validation of the request will check that an URI is
        present inside the request. This is a flag forwarded to _get_validated_request() method.
        :return: request, service_name, work_in_gray, bounding_box
        """
        request = GenericEstimationController._get_validated_request(should_have_uri=should_have_uri)
        service_name = request.get('service', 'default')
        work_in_gray = request.get('work_in_gray', "true") == "true"
        bounding_box = request.get('bounding_box', None)

        return request, service_name, work_in_gray, bounding_box

    @staticmethod
    def _crop_by_bounding_box(image, bounding_box_text=None):
        """
        Crops the image by the specified bounding box. If no bounding box is specified, the image
        will be returned without any filtering.
        :param image: image to crop.
        :param bounding_box_text: bounding box to crop by. None to disable cropping.
        :return: the image filtered or the original image in case the bounding box is None.
        """

        if bounding_box_text is not None:
            try:
                bounding_box_text = BoundingBox.from_string(bounding_box_text)
                bounding_box_text.fit_in_size(image.get_size())
                image = image.crop_image(bounding_box_text, "Cropped by boundingbox")

            except Exception as ex:
                raise InvalidRequest("The bounding box format is not valid. "
                                     "Format must be: bounding_box=x,y,width,height")

        return image

    def _generic_request(self, image, bounding_box, service):
        """
        Generic request of the controller. Common actions among different requests will converge here.
        :param image: image to process. It must be already loaded.
        :param bounding_box: bounding box to crop the image by. None to disable cropping.
        :param service: service to process the resource.
        :return: returns the full response object built with the result of the process.
        """

        # If a bounding-box analysis is requested, we crop the image into the specified bounding box:
        image = self._crop_by_bounding_box(image, bounding_box)

        # This will block the request until the resource is ready.
        result = service.append_request(image).get_resource()

        estimation_result = self._retrieve_result_metadata(result)[0]

        return jsonify(estimation_result.to_dict())
