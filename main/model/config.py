#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import os

from main.model.resource.image import Image

__author__ = "Ivan de Paz Centeno"

DEFAULT_CONFIG_FILE = 'main/etc/module.cfg'

# This dict is filled by the algorithms loaded by the APP.
# It must contain the algorithm code name, which contains the type, subtype and extra data.
AVAILABLE_ALGORITHMS = {}
SERVICE_PROTOTYPE_BY_RESOURCE_TYPE = {}


def fix_working_dir():
    """
    Fixes the working directory. Each process/thread in the project should call
    this method at the very beginning to update its working directory.
    """
    # Fix the working dir
    os.chdir("{}/../..".format(os.path.dirname(__file__)))


class Config(object):
    """
    Stores options read from the configuration file.
    For example, the IP, Port of the web API REST and each of the services available for the controllers.

    A service is just a pool of algorithms configurable in some way, for example
    setting the number of workers for the pool or the usage of GPU or not.
    """

    def __init__(self, config_file=DEFAULT_CONFIG_FILE, ignore_service_when_algorithm_not_available=False):
        """
        Initializes the object with the config file loaded into memory.
        It will check that the syntax of the file is correct.

        :param ignore_service_when_algorithm_not_available: if set to true, when an algorithm
                is not available, the service that owns it will be discarded.
                Otherwise an exception will be thrown.
        """
        settings_loader = configparser.ConfigParser()
        settings_loader._interpolation = configparser.ExtendedInterpolation()
        _ = settings_loader.read(config_file, encoding="UTF-8")

        image = Image(config_file)

        # Stores each of the services definition by section name ( algorithm, name, description, workers, GPU )
        self.services_definition = {}
        self.web_app_definition = {'ip': '0.0.0.0', 'port': 1025}

        self._build_available_services_definition(settings_loader)
        self._check_definitions_correctness(ignore_service_when_algorithm_not_available)

    def _build_available_services_definition(self, settings_loader):
        """
        Builds a memory-version of the services definition.
        """

        # There should be a main section called APP with the IP and PORT desired for the REST API.
        for service_section in settings_loader.sections():

            if service_section.upper() == "APP":
                print("Loaded APP config.")
                self.web_app_definition = {
                    'ip': settings_loader.get(service_section, "IP", fallback="localhost"),
                    'port': settings_loader.getint(service_section, "PORT", fallback=1025),
                }

            else:
                self.services_definition[service_section] = {
                    'algorithm': settings_loader.get(service_section, "ALGORITHM"),
                    'public_name': settings_loader.get(service_section, "PUBLIC_NAME"),
                    'description': settings_loader.get(service_section, "DESCRIPTION"),
                    'use_gpu': settings_loader.getint(service_section, "USE_GPU", fallback=-1),
                    'workers': settings_loader.getint(service_section, "WORKERS"),
                    'default': settings_loader.getboolean(service_section, "DEFAULT", fallback=False),
                }

                with_gpu = self.services_definition[service_section]['use_gpu']
                workers = self.services_definition[service_section]['workers']

                if with_gpu == -1:
                    extra_info = "mapped into CPU"
                else:
                    extra_info = "mapped into GPU with index {}".format(with_gpu)

                print("Loaded service \"{}\" with {} workers {}.".format(service_section, workers, extra_info))

    def _check_definitions_correctness(self, ignore_service_when_algorithm_when_not_available):
        """
        Checks the correctness of the services definition.
        If the correctness is incorrect, an exception is thrown.
        :param ignore_service_when_algorithm_when_not_available: if set to true, when an algorithm
                is not available, the service that owns it will be discarded.
                Otherwise an exception will be thrown.
        """
        services_keys_to_ignore = []

        for service_name in self.services_definition:
            definition = self.services_definition[service_name]

            if 'algorithm' not in definition:
                raise Exception("Algorithm must be defined in the section {}".format(service_name))

            if definition['algorithm'] not in AVAILABLE_ALGORITHMS:
                if ignore_service_when_algorithm_when_not_available:
                    print("Warning: algorithm {} not available.".format(definition['algorithm']))
                    services_keys_to_ignore.append(service_name)
                else:
                    raise Exception("Algorithm {} not available.".format(definition['algorithm']))

        # If there are keys to ignore, let's remove them from the dictionary of definitions.
        for service_name in services_keys_to_ignore:
            del self.services_definition[service_name]

    def get_services_definition(self):
        """
        Getter for the services definition.
        """
        return self.services_definition

    def get_webapp_definition(self):
        """
        Getter for the web app definition.
        """
        return self.web_app_definition