#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from threading import Thread, Lock
from main.services.status import SERVICE_STOPPED, SERVICE_RUNNING, SERVICE_STOPPING


__author__ = 'Iván de Paz Centeno'


class Service(object):
    """
    Allows the process of something in background. It has states, and basic methods of start, stop and join.
    """

    def __init__(self):
        """
        Initialization of the service.
        """
        self.__do_stop = False
        self.__status = SERVICE_STOPPED
        self.lock = Lock()

    def __reset_thread__(self):
        """
        Resets the internal worker thread.
        :return:
        """
        self.worker_thread = Thread(target=self.__internal_thread__)

    def __set_status__(self, status_value):
        """
        Sets thread-safely the value of the status.
        Use this method instead of accessing the __status attribute directly

        :param value: SERVICE_RUNNING or SERVICE_STOPPED
        """
        with self.lock:
            self.__status = status_value

    def __get_status__(self):
        """
        Retrieves the status value thread-safely.
        Use this method instead of accessing the __status attribute directly.

        :return: Status of the service. May be SERVICE_RUNNING or SERVICE_STOPPED
        """
        with self.lock:
            status = self.__status

        return status

    def get_status(self):
        """
        :return: status code of the service. You can convert this code to a string representation with
        status_code_to_name() function from status.py
        """
        return self.__get_status__()

    def start(self):
        """
        Starts the service in background.

        """
        if self.get_status() >= SERVICE_RUNNING:
            return

        self.__reset_thread__()
        self.__set_status__(SERVICE_RUNNING)
        self.worker_thread.start()

    def stop(self, wait_for_finish=True):
        """
        Stops the service from working on background.

        :type wait_for_finish: bool   specify if the current thread must wait for the service to
        finish or not.
        """

        if self.get_status() <= SERVICE_STOPPED:
            return

        self.__set_status__(SERVICE_STOPPING)

        if wait_for_finish:
            self.worker_thread.join()

    def __internal_thread__(self):
        """
        Internal backgrounded code. Override this method with your own.
        Ensure that the 'status' variable is checked to exit the loop when it is SERVICE_STOPPING.
        Do not forget to invoke this super method in the child at the end of your method.
        :return: None
        """

        self.__set_status__(SERVICE_STOPPED)
        return None
