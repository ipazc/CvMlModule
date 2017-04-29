#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy
from main.controllers.controller import Controller
from main.model.resource.image import Image


__author__ = "Ivan de Paz Centeno"


class ImageController(Controller):
    """
    Generic controller super class for images-based ones.
    Inherit it to build a controller for images.
    """

    @staticmethod
    def _build_image_from_content(content, as_gray=True):
        """
        Builds the image resource from the image raw content.
        :param content: Image content to be loaded.
        :return: Image resource containing the content of the image parsed by OpenCV loader.
        """

        if as_gray:
            read_flag = cv2.IMREAD_GRAYSCALE
        else:
            read_flag = cv2.IMREAD_COLOR

        nparr = numpy.fromstring(content, numpy.uint8)
        content = cv2.imdecode(nparr, read_flag)

        return Image(uri="memorycontent", image_id="memory", blob_content=content)

    @staticmethod
    def _build_image_from_uri(uri):
        """
        Builds an image instance from a URI.
        This method won't make checks or loads from the specified URI.
        :param uri: uri to create the image.
        :return: image instance wrapping the URI.
        """
        return Image(uri=uri)
