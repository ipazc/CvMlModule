#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import sleep
from main.model.config import SERVICE_PROTOTYPE_BY_RESOURCE_TYPE
from main.model.resource.image import Image
from main.model.resource.resource_promise import ResourcePromise
from main.services.pool.algorithm_pool import AlgorithmPool
from main.services.service import Service
from main.services.status import SERVICE_RUNNING

__author__ = 'Iván de Paz Centeno'


class ImageAlgorithmService(Service, AlgorithmPool):
    """
    Service for algorithms based on Images.
    """

    def __init__(self, algorithm, pool_limit=None, use_gpu=-1):
        """
        Initializer of the service.
        :param algorithm: algorithm prototype. This algorithm class prototype will be instantiated
                          once per process.
        :param pool_limit: number of processes for the pool.
        :param use_gpu: -1 to use CPU; 0 to use GPU0; 1 to use GPU1; ...
        """
        Service.__init__(self)
        AlgorithmPool.__init__(self, algorithm, pool_limit, use_gpu)
        # We map resource to promise
        self.promises_dict = {}

    def append_request(self, resource, extra_data=None):
        """
        Appends the resource into the queue of the pool.
        :param resource: resource to process.
        :param extra_data: anything else to pass to the processor.
        :return : promise object for the result
        """

        # Imagine that multiple requests for the same image content are demanded.
        # It could be easier if we return the same ResourcePromise for all of them, isn't it?
        # Since the promise makes the thread to wait until the resource is ready,
        # so we avoid to compute same image multiple times.
        duplicated = False

        with self.lock:
            # If a similar resource is being processed, we don't queue it.
            # Instead, we take it from the queue.
            if resource.md5hash() in self.promises_dict:
                result_promise = self.promises_dict[resource.md5hash()]
                duplicated = True
            else:
                result_promise = ResourcePromise(self.manager)
                self.promises_dict[resource.md5hash()] = result_promise

        if not duplicated:
            self._queue_resource(resource, extra_data)
            self._process_queue()

        return result_promise

    def __internal_thread__(self):
        """
        Internal thread of the service.
        :return:
        """
        while self.get_status() >= SERVICE_RUNNING:
            sleep(0.5)  # TODO: change the sleep with a lock or event.

        AlgorithmPool.terminate(self)
        Service.__internal_thread__(self)

    def process_finished(self, wrapped_result):
        """
        Method invoked when the process of the algorithm finished the resource.
        :param wrapped_result: parameters of the result (the initial resource, the result and some extra data).
        :return:
        """

        try:
            resource = wrapped_result[0]
            result = wrapped_result[1][0]

            promise = None
            with self.lock:
                if resource.md5hash() not in self.promises_dict:
                    raise Exception("Error: the resource does not have a promise associated. Request discarded.")

                promise = self.promises_dict[resource.md5hash()]
                del self.promises_dict[resource.md5hash()]

            promise.set_resource(result)

        except Exception as ex:
            print(ex)

        # invocation of super.
        AlgorithmPool.process_finished(self, wrapped_result)

        self._process_queue()

    @staticmethod
    def get_resource_type():
        """
        :return: the resource type of the algorithms that this service manages.
        """
        return Image


# We register here the algorithm service by the resource type it handles.
# Remember that this is the prototype of the AlgorithmService, and this is going to be used
# to create AlgorithmServices for the given resource type.
SERVICE_PROTOTYPE_BY_RESOURCE_TYPE[ImageAlgorithmService.get_resource_type()] = ImageAlgorithmService
