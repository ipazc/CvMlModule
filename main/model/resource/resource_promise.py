#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = "Ivan de Paz Centeno"


class ResourcePromise(object):
    """
    Promise object
    Wraps a resource in order to allow storage of a resource between different threads.
    Also, it allows to wait for the resource to be ready.
    """

    def __init__(self, multithread_manager):
        """
        Initializes the resource container.
        """

        self.resource = None
        self.lock = multithread_manager.Lock()
        self.event = multithread_manager.Event()

    def set_resource(self, resource):
        """
        Setter for the resource.
        """

        with self.lock:
            self.resource = resource

        self.event.set()

    def get_resource(self):
        """
        Getter for the resource. It will wait until the resource is ready.
        :return: Resource object.
        """

        self.event.wait()

        with self.lock:
            resource = self.resource

        return resource
