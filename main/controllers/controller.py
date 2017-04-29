#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
from functools import partial
from multiprocessing import Lock
from flask import jsonify, request
from main.exceptions.invalid_request import InvalidRequest
from main.model.config import AVAILABLE_ALGORITHMS


__author__ = "Ivan de Paz Centeno"


def route(*args, **kwargs):
    """
    Decorator proxy for @self.flask_web_app.route.
    Since we cannot use "self" while calling the decorator, proxying it is a good solution.
    :param args:
    :param kwargs:
    :return:
    """
    def decorator1(func):

        def decorator2(self):

            app = self.flask_web_app
            # At this level we have the app defined, extracted from self.

            partial_func = partial(func, self)
            app.add_url_rule(*(args + ("{}.{}".format(self.__class__.__name__, func.__name__), partial_func)), **kwargs)

        return decorator2

    return decorator1


def error_handler(*args, **kwargs):
    """
    Decorator proxy for @self.flask_web_app.errorhandler.

    Since we cannot use "self" while calling the decorator, proxying it is a good solution.
    :param args:
    :param kwargs:
    :return:
    """

    def decorator1(func):
        def decorator2(self):
            app = self.flask_web_app

            # At this level we have the app defined, extracted from self.

            partial_func = partial(func, self)

            app.register_error_handler(*(args + (partial_func,)))

        return decorator2

    return decorator1


class Controller(object):
    """
    Generic controller super class.
    Inherit it to build a controller.
    """

    def __init__(self, flask_web_app, available_services, config, controller_type, controller_subtype):
        """
        Constructor of the controller.
        :param flask_web_app: flask app object.
        :param available_services:  dict of services available ("service_name" -> service_object).
        :param config:  config object with all the information regarding the definition of services.
        :param controller_type:  string containing the type of the controller ("DETECTION", "ESTIMATION",
                                                                               "CLASSIFICATION", "SEGMENTATION",...)
        :param controller_subtype:  string containing the subtype of the controller ("FACE", "AGE", "GENDER",
                                                                                     "SKIN", ...)
        """
        self.lock = Lock()
        self.flask_web_app = flask_web_app
        self.available_services = available_services
        self.config = config
        self.type = controller_type
        self.subtype = controller_subtype
        self.default_service = self._find_default_service()
        self.exposed_methods = [
            self.handle_invalid_request
        ]

    def _init_exposed_methods(self):
        """
        Initializes the exposed methods to be API-REST callable.
        """
        [exposed_method() for exposed_method in self.exposed_methods]

    def _find_default_service(self):
        """
        Searchs for the default service for this controller from the available services.
        It is going to be the first with the 'default' flag set to True.

        If no default service is found, it will get one randomly.
        :return: The default service for this controller.
        """
        default_service = None

        for service_name in self.available_services:
            default_service = self.available_services[service_name]

            if self.config.get_services_definition()[service_name]['default']:
                break

        return default_service

    def get_available_services(self):
        """
        Override this method adding the route() decorator set to the one that suits the controller.
        :return: the available algorithms for the controller.
        """

        services_definition = self.config.get_services_definition()

        result = []

        for service_name in self.available_services:
            description = services_definition[service_name]['description']
            public_name = services_definition[service_name]['public_name']
            resource_type = self.available_services[service_name].get_resource_type()
            detection_type = AVAILABLE_ALGORITHMS[services_definition[service_name]['algorithm']]['detection_type']

            result.append({
                'name': service_name,
                'public_name': public_name,
                'description': description,
                'type': self.type,
                'subtype': self.subtype,
                'resource_type': resource_type.__name__,
                'detection_type': detection_type.__name__
            })

        return {'available_services': result}

    def _get_most_suitable_service(self, service_name):
        """
        Searchs for the most suitable service for the requested service name.
        If a service name is provided but no services matches it, it will return an InvalidRequest exception.
        :param service_name: name of the service to match. If it is 'default', it will fall back to the
                             default service for this controller.
        :return: the most suitable service for the service name.
        """

        if service_name == "default":
            service = self.default_service

        else:
            if service_name not in self.available_services:
                raise InvalidRequest("Service name '{}' not valid.".format(service_name))

            service = self.available_services[service_name]

        return service

    @staticmethod
    def _retrieve_result_metadata(resource_result):
        """
        Applies the algorithm of the service to the resource and gets the result.
        :param resource_result: Result of the process from a service.
        :return: the metadata result of the process.
        """

        if resource_result.get_uri() == 'error':
            raise InvalidRequest("Content of file not valid: {}".format(resource_result.get_id()))

        return resource_result.get_metadata()

    @staticmethod
    def _get_validated_request(should_have_uri=False):
        """
        Validates the request to fit a basic protocol. If the request does not fit the protocol, it will raise
        an exception.
        Override this method to change or extend the default validation of the incoming request.

        :param should_have_uri: flag to specify if the request should contain an URI or not.
        :return: the request validated.
        """

        request_args = request.args

        if should_have_uri and request_args.get("uri", '') == '':
            raise InvalidRequest("Request URI is missing.")

        return request_args

    @staticmethod
    def _get_raw_content_validated(is_base64=False):
        """
        Validates the content of the request to ensure that it is readable and returns it.
        Override this method to change or extend the default validation of the incoming request.

        :param is_base64:   specifies if the content is encoded in base64 or not. In case it is encoded, it will try
        to decode it.
        :return:    the raw content (decoded content in case of base64)
        """
        content = request.stream.read()

        if len(content) == 0:
            raise InvalidRequest("Request without content.")

        if is_base64:
            try:
                content = base64.b64decode(content)
            except Exception as ex:
                raise InvalidRequest("Content is not valid Base64.")

        return content

    @error_handler(InvalidRequest)
    def handle_invalid_request(self, error):
        """
        Handles the invalid request error in order to formally notify the appropriate code.
        :param error: dict containing the explanation of the error.
        :return:
        """
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    def release_services(self, wait_for_close=True):
        """
        Releases the controller's services
        """

        for service in self.available_services.values():
            service.stop(wait_for_finish=wait_for_close)

        self.available_services.clear()

    def __del__(self):
        """
        On destruction of the object, all the services are released.
        """

        self.release_services()
