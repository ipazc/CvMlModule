#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from main.model.normalizer.normalizer import Normalizer
from main.model.tools.boundingbox import BoundingBox

__author__ = 'Iván de Paz Centeno'


class ProportionSizeNormalizer(Normalizer):
    """
    Resizes a bounding box to a specific size.
    """

    def __init__(self, proportion_width, proportion_height):
        """
        Constructor for the size normalizer.
        :param proportion_width: pixels proportion to apply (multiply) to the width.
        :param proportion_height: pixels proportion to apply (multiply) to the height.
        """
        self.proportion_width = proportion_width
        self.proportion_height = proportion_height

    def apply(self, bounding_box):
        """
        Applies the resize normalizer to the specified bounding box.
        :param bounding_box: Bounding box to resize.
        :return: A bounding box clone resized
        """

        box = bounding_box.get_box()

        return BoundingBox(
            int(box[0] * self.proportion_width),
            int(box[1] * self.proportion_height),
            int(box[2] * self.proportion_width),
            int(box[3] * self.proportion_height)
        )
