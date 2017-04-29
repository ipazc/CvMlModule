#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import caffe
import main.model.predictor.overriden_caffe.classifier as overriden_caffe
from main.model.stdfile_redirector import stdfile_redirector
from skimage import img_as_float

__author__ = 'Iván de Paz Centeno'


class CNNCaffePredictor(object):
    """
    Allows to perform predictions based on a mean file, a pretrained model and a network topology model.
     Employs the Caffe framework to run a CNN and is capable of running this task in CPU or GPU if desired.
    """
    def __init__(self, mean_filename, pretrained_net_model, net_model, use_gpu, tags=None):
        """
        Constructor of the class.
        :param mean_filename: path to the file that contains the mean file for the model.
        :param pretrained_net_model: path to the file that contains the network model values.
        :param net_model: path to the file that contains the topology of the network.
        :param use_gpu: specifies which GPU should be used for this predictor. set to -1 to use the CPU,
        0 to use the first GPU, 1 to use the second GPU, and so on.
        :param tags: tag list for the prediction result. It must contain as many items
         as output in the network.
        """
        self.mean_filename = mean_filename
        self.pretrained_net_model = pretrained_net_model
        self.net_model = net_model

        if not tags:
            tags = []

        self.tags = tags

        if use_gpu > -1:
            caffe.set_device(use_gpu)
            caffe.set_mode_gpu()

        self.__load_mean_file__()
        self.__load_classifier__()

    def __load_mean_file__(self):
        """
        Loads the mean file to the predictor.
        """
        with open(self.mean_filename, "rb") as mean_file:
             proto_data = mean_file.read()

        blob_proto = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        self.mean = caffe.io.blobproto_to_array(blob_proto)[0]

    def __load_classifier__(self, channel_swap=(2, 1, 0), raw_scale=255, image_dims=(256, 256)):
        """
        Loads the classifier from the Caffe library
        :param channel_swap: injected parameter for the classifier.
        :param raw_scale: injected parameter for the classifier.
        :param image_dims: injected parameter for the classifier.
        :return: classifier instance with the network built.
        """
        with stdfile_redirector():  # We need to hide stderr.
            self.classifier = overriden_caffe.Classifier(self.net_model, self.pretrained_net_model,
                                               mean=self.mean,
                                               channel_swap=channel_swap,
                                               raw_scale=raw_scale, image_dims=image_dims)

    def __predict_image__(self, image_content):
        """
        Retrieves the tag index for the prediction (the argmax of the latest layer of the CNN after it is fed).
        :param image_content: content of the image (numpy array)
        :return: prediction in form of index of tags.
        """

        # The prediction works with floats
        input_image = img_as_float(image_content)

        with stdfile_redirector():

            # Since the caffe predictor output layer are multiple nodes,
            # we take prediction from the node whose value is MAX.
            # In the case of the age, the *index* of that node is the predicted age.
            # In the case of the gender, the *index* of that node is the predicted gender.
            prediction = self.classifier.predict([input_image])[0].argmax()

        return prediction

    def predict_image(self, image_content):
        """
        Predicts the given content into one of the defined tags. If no tags are given, the
        argmax of the result (the neuron index with highest value) will be returned.
        :param image_content: image content to predict.
        :return: tag name for the given content or the argmax in case tags are not provided.
        """

        prediction = self.__predict_image__(image_content)

        if len(self.tags) > 0:
            result = self.tags[self.__predict_image__(image_content)]
        else:
            result = prediction

        return result
