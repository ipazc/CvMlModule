#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from main.model.algorithm.detection.face.mtcnn.mtcnn_face_detector import MTCNNFaceDetector
from main.model.normalizer.boundingbox.proportion_size_normalizer import ProportionSizeNormalizer
from main.model.normalizer.image.absolute_size_normalizer import AbsoluteSizeNormalizer
from main.model.tools.boundingbox import BoundingBox
from main.model.config import AVAILABLE_ALGORITHMS
from main.model.algorithm.image_algorithm import ImageAlgorithm


__author__ = 'Iván de Paz Centeno'

# Pixels that the images are allowed to have(at max).
NORMALIZE_IMAGES_SIZE = (1024, 1024)

class MTCNNFaceDetectionAlgorithm(ImageAlgorithm):
    """
    Algorithm for detection of faces based on CNN.
    @article{7553523,
        author={K. Zhang and Z. Zhang and Z. Li and Y. Qiao},
        journal={IEEE Signal Processing Letters},
        title={Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks},
        year={2016},
        volume={23},
        number={10},
        pages={1499-1503},
        keywords={Benchmark testing;Computer architecture;Convolution;Detectors;Face;Face detection;Training;Cascaded convolutional neural network (CNN);face alignment;face detection},
        doi={10.1109/LSP.2016.2603342},
        ISSN={1070-9908},
        month={Oct}
    }
    """

    def __init__(self, use_gpu=-1):
        """
        Initializes the algorithm.
        :param use_gpu: parameter to set the GPU usage for this algorithm.
        The number represents the index of the GPU in the machine, being -1 the CPU.
        WARNING: This algorithm does not support the usage of GPU yet.
        """

        ImageAlgorithm.__init__(self, MTCNNFaceDetectionAlgorithm.__name__,
                                "MT Face detection Algorithm based on CNN (Caffe).")

        self.detector = MTCNNFaceDetector(use_gpu=use_gpu)
        self.size_normalizer = AbsoluteSizeNormalizer(NORMALIZE_IMAGES_SIZE[0], NORMALIZE_IMAGES_SIZE[1],
                                                      keep_aspect_ratio=True)

    def _process_resource(self, image):
        """
        Processes the specified image in order to get the bounding boxes for the faces.
        :param image: image resource pointing to a valid URI or containing the image content.
                    If the image is not loaded but is pointing to a valid URI, this method
                    will try to load the image from the URI in grayscale.
        :return: an array of bounding boxes
        """
        #  This is a required step because MTCNN has a limitation on the size it can process.
        # 1024x1024 is an affordable size for MTCNN.

        if image.get_size() > NORMALIZE_IMAGES_SIZE:
            normalized_image = self.size_normalizer.apply(image)
            proportions = [ x/y if y else 1 for x, y in zip(image.get_size(), normalized_image.get_size()) ]
            proportion_bbox_normalizer = ProportionSizeNormalizer(*proportions)

        else:
            normalized_image = image
            proportion_bbox_normalizer = None

        image_content = self._get_loaded_image_content(normalized_image, as_gray=False)

        detections, points = self.detector.detect_faces(image_content)

        metadata_content = []

        for detection in detections:
            raw_bbox = list([int(x) for x in detection[:4]])

            normalized_bounding_box = BoundingBox(raw_bbox[0], raw_bbox[1],
                                                  raw_bbox[2] - raw_bbox[0],
                                                  raw_bbox[3] - raw_bbox[1])

            # Bounding boxes are relative to the normalized image. We need to resize them back to the original size
            if proportion_bbox_normalizer is not None:
                bounding_box = proportion_bbox_normalizer.apply(normalized_bounding_box)
            else:
                bounding_box = normalized_bounding_box

            bounding_box.fit_in_size(image.get_size())
            metadata_content.append(bounding_box)

        return metadata_content


# It needs to be registered here.
AVAILABLE_ALGORITHMS[MTCNNFaceDetectionAlgorithm.__name__] = {
    'prototype': MTCNNFaceDetectionAlgorithm,
    'resource_type': MTCNNFaceDetectionAlgorithm.kind_of_resource(),
    'type': 'DETECTION',
    'subtype': 'FACE',
    'detection_type': BoundingBox
}
