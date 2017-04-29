#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import signal
from multiprocessing import Pool, Manager
from queue import Empty
from main.model.resource.resource import Resource

__author__ = 'Iván de Paz Centeno'


algorithm_detector = None


def process(queue_element):
    """
    Processes the queued element applying the algorithm to the input.
    :param queue_element: a list with [resource, resource_result_container and extra_data]
    :return:
    """
    global algorithm_detector
    algorithm = algorithm_detector
    resource = queue_element[0]
    extra_data = queue_element[1]

    try:
        if not algorithm.is_resource_processable(resource):
            raise Exception("Resource type is not admited by the algorithm.")

        if not resource.is_loaded():
            raise Exception("Resource was empty. Couldn't perform the analysis on an empty resource.")

        result = algorithm.process_resource(resource)

    except Exception as ex:
        result = (Resource(uri="error", res_id=ex.__str__()), 0)

    return [resource, result, extra_data]


class AlgorithmPool(object):
    """
    Represents a pool of algorithms of a specified type.
    It allows the parallel process of resources, with or without GPU.

    Override process_finished to retrieve the result.
    """
    def __init__(self, algorithm, pool_limit, use_gpu=-1):
        self.algorithm = algorithm
        self.manager = Manager()
        self.processing_queue = self.manager.Queue()
        self.algorithm_detectors = {}

        if not pool_limit or pool_limit == "auto":
            pool_limit = None
        else:
            pool_limit = int(pool_limit)

        self.pool = Pool(processes=pool_limit, initializer=self.__init_pool_worker__, initargs=(algorithm, use_gpu))

        self.algorithms_free = self.pool._processes

    @staticmethod
    def __init_pool_worker__(algorithm, use_gpu):
        """
        Initializes the worker resources (on its own context)
        :param algorithm: algorithm prototype in order to instantiate it
        :param use_gpu: flag to specify the GPU to use (0 = GPU0, 1 = GPU1, ... -1 = CPU)
        """

        global algorithm_detector
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        algorithm_detector = algorithm(use_gpu=use_gpu)

    def _queue_resource(self, resource, extra_data):
        """
        Queues the specified resource in order for the pool to process it when a process is free.
        :param resource: resource to process.
        :param extra_data:
        :return:
        """
        self.processing_queue.put([resource, extra_data])

    def _process_queue(self):
        """
        Processes the queue until it is clean.
        """
        queue_empty = False

        while self.algorithms_free > 0 and not queue_empty:
            try:
                queue_element = self.processing_queue.get(False)
                self.algorithms_free -= 1
                result = self.pool.apply_async(process, args=(queue_element,), callback=self.process_finished)

            except Empty as emp:
                queue_empty = True

    def process_finished(self, wrapped_result):
        """
        When the process is finished this method is invoked. Override it to access to the result.
        *NOTE:* Do not forget to call this super method at the very beginning of your method!!
        :param wrapped_result:
        """
        self.algorithms_free += 1

        # Override this method
        return None

    def terminate(self):
        """
        Releases the pool resources.
        """
        self.pool.terminate()
        self.pool.join()

    def get_algorithm_proto(self):
        """
        :return: the algorithm proto for the current pool.
        """
        return self.algorithm

    def __del__(self):
        # WARNING: it may require a check to see if terminate() was already called before.
        self.terminate()