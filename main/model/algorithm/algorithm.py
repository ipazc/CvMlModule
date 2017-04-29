#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from timeit import default_timer as timer
from main.model.resource.resource import Resource


__author__ = 'Iván de Paz Centeno'


class Algorithm:
    def __init__(self, name, description):
        """
        Instantiates an algorithm.

        :param name:    name of the algorithm. Must be short, lower case and underscores "_" instead of spaces " ".
        :param description:     Full description of the algorithm. Useful for reports after evaluation.
        """

        self.description = description
        self.name = name

    def get_description(self):
        return self.description

    def get_name(self):
        return self.name

    def process_resource(self, resource):
        """
        Applies the algorithm to the specific resource and returns a result.

        :param resource: resource to check. It must be a class inherited from Resource.

        :return: a result of the algorithm and the time spent in nanoseconds precision of the process.
        Usually, the result of the algorithm is another resource with its metadata updated and the URI set to the
        same path with the algorithm name appended.
        """

        assert self._process_resource is not None, "A virtual algorithm can't process a resource."

        start_time = timer()

        # Override the method _process_resource with the code of the algorithm.
        metadata_content = self._process_resource(resource)

        new_uri = self.__generate_new_uri__(resource)
        result = self.kind_of_resource()(uri=new_uri, metadata=metadata_content)

        time_spent = timer() - start_time

        return result, time_spent

    def __generate_new_uri__(self, resource):
        path, filename = os.path.split(resource.get_uri())
        new_path = path+"_"+self.get_name()

        return os.path.join(new_path, filename)

    def is_resource_processable(self, resource):
        """
        Determines if a resource is procesable by this algorithm or not.
        This is a virtual methods. Extend it to add compatibility for a specific resource to your algorithm
        :param resource: Resource to check compaitibility with the algorithm
        :return: True if compatible. False othwerise.
        """
        return False

    def __str__(self):
        """
        :return: string representation of the algorithm
        """
        return "[Algorithm {}: \"{}\"; admits resources of type \"{}\"]".format(self.name, self.description,
                                                                                self.kind_of_resource())

    @staticmethod
    def kind_of_resource():
        """
        Returns the kind of resource of this algorithm
        :return:
        """
        return Resource
