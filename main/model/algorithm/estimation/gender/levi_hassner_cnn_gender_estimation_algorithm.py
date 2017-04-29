#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from main.model.algorithm.estimation.caffe_image_cnn_generic_estimation_algorithm import \
    CaffeImageCNNGenericEstimationAlgorithm
from main.model.predictor.cnn_caffe_predictor import CNNCaffePredictor
from main.model.tools.gender import Gender, GENDER_MALE, GENDER_FEMALE, GENDER_UNKNOWN
from main.model.config import AVAILABLE_ALGORITHMS


__author__ = 'Iván de Paz Centeno'

MEAN_FILENAME = "main/data/caffe/levihassner/mean.binaryproto"
PRETRAINED_NET_MODEL = "main/data/caffe/levihassner/gendermodels/gender_net.caffemodel"
NET_MODEL = "main/data/caffe/levihassner/gendermodels/deploy_gender.prototxt"
TAGS = [Gender(GENDER_MALE),
        Gender(GENDER_FEMALE),
        Gender(GENDER_UNKNOWN)]


class LeviHassnerCNNGenderEstimationAlgorithm(CaffeImageCNNGenericEstimationAlgorithm):
    """
    Algorithm for estimation of genders of faces based on CNN implementation in Caffe from
    Gil Levi and Tal Hassner.
    """

    def __init__(self, use_gpu=-1):
        """
        Initializes the algorithm.
        :param use_gpu: parameter to set the GPU usage for this algorithm.
        The number represents the index of the GPU in the machine, being -1 the CPU.
        """

        CaffeImageCNNGenericEstimationAlgorithm.__init__(self, LeviHassnerCNNGenderEstimationAlgorithm.__name__,
                                                         "CNN based Gender estimation, from Levi and Hassner work "
                                                         "(ADIENCE), over Caffe")

        self.estimator = CNNCaffePredictor(MEAN_FILENAME, PRETRAINED_NET_MODEL, NET_MODEL, use_gpu, TAGS)


# It needs to be registered here.
AVAILABLE_ALGORITHMS[LeviHassnerCNNGenderEstimationAlgorithm.__name__] = {
    'prototype': LeviHassnerCNNGenderEstimationAlgorithm,
    'resource_type': LeviHassnerCNNGenderEstimationAlgorithm.kind_of_resource(),
    'type': 'ESTIMATION',
    'subtype': 'GENDER',
    'detection_type': Gender
}
