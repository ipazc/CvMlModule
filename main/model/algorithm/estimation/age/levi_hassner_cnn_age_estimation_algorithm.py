#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from main.model.algorithm.estimation.caffe_image_cnn_generic_estimation_algorithm import \
    CaffeImageCNNGenericEstimationAlgorithm
from main.model.predictor.cnn_caffe_predictor import CNNCaffePredictor
from main.model.tools.age_range import AgeRange
from main.model.config import AVAILABLE_ALGORITHMS


__author__ = 'Iván de Paz Centeno'

MEAN_FILENAME = "main/data/caffe/levihassner/mean.binaryproto"
PRETRAINED_NET_MODEL = "main/data/caffe/levihassner/agemodels/age_net.caffemodel"
NET_MODEL = "main/data/caffe/levihassner/agemodels/deploy_age.prototxt"
TAGS = [AgeRange(0, 2),
        AgeRange(4, 6),
        AgeRange(8, 12),
        AgeRange(15, 20),
        AgeRange(25, 32),
        AgeRange(38, 43),
        AgeRange(48, 53),
        AgeRange(60, 100)]


class LeviHassnerCNNAgeEstimationAlgorithm(CaffeImageCNNGenericEstimationAlgorithm):
    """
    Algorithm for estimation of ages of faces based on CNN implementation in Caffe from
    Gil Levi and Tal Hassner.
    """

    def __init__(self, use_gpu=-1):
        """
        Initializes the algorithm.
        :param use_gpu: parameter to set the GPU usage for this algorithm.
        The number represents the index of the GPU in the machine, being -1 the CPU.
        """

        CaffeImageCNNGenericEstimationAlgorithm.__init__(self, LeviHassnerCNNAgeEstimationAlgorithm.__name__,
                                                         "CNN based Age estimation, from Levi and Hassner work "
                                                         "(ADIENCE), over Caffe")

        self.estimator = CNNCaffePredictor(MEAN_FILENAME, PRETRAINED_NET_MODEL, NET_MODEL, use_gpu, TAGS)


# It needs to be registered here.
AVAILABLE_ALGORITHMS[LeviHassnerCNNAgeEstimationAlgorithm.__name__] = {
    'prototype': LeviHassnerCNNAgeEstimationAlgorithm,
    'resource_type': LeviHassnerCNNAgeEstimationAlgorithm.kind_of_resource(),
    'type': 'ESTIMATION',
    'subtype': 'AGE',
    'detection_type': AgeRange
}
