#!/usr/bin/env python
# -*- coding: utf-8 -*-

from main.controllers.ensemble_requests.face_ensemble import FaceEnsembleController
from main.controllers.estimation_requests.age_estimation import AgeEstimationController
from main.controllers.detection_requests.face_detection import FaceDetectionController
from main.controllers.estimation_requests.gender_estimation import GenderEstimationController
from main.model.config import AVAILABLE_ALGORITHMS, SERVICE_PROTOTYPE_BY_RESOURCE_TYPE

__author__ = "Ivan de Paz Centeno"


CONTROLLERS_LIST = {
    'face_detection_controller': FaceDetectionController,
    'age_estimation_controller': AgeEstimationController,
    'gender_estimation_controller': GenderEstimationController,
    'face_ensemble_controller': FaceEnsembleController,
}


class ControllerFactory(object):
    """
    Factory for controllers.
    Eases the creation of controllers by injecting common dependencies, like the services definition
    and the app handler.
    """

    def __init__(self, flask_app, config):
        """
        Initializes the factory with minimum parameters.
        :param flask_app: application of Flask to handle the requests.
        :param config: configuration object containing all the definitions of services.
        """
        self.flask_app = flask_app
        self.config = config
        self.controllers = {}
        self.available_services = {}
        self._build_services_from_definitions()

        # This dict acts like the global CONTROLLERS_LIST, but
        # proxying the instantation with the factory method.
        # This is specially useful for ensemble controllers that depends
        # on atomic controllers.
        self.controllers_creation_method = {
            'face_detection_controller': self.face_detection_controller,
            'age_estimation_controller': self.age_estimation_controller,
            'gender_estimation_controller': self.gender_estimation_controller,
            'face_ensemble_controller': self.face_ensemble_controller,
        }

    def _build_services_from_definitions(self):
        """
        Builds the services given a definition list.
        The services are stored inside self.available_services.
        These services will be injected into the controllers.
        """
        definitions = self.config.get_services_definition()

        for service_definition_name in definitions:

            service_definition = definitions[service_definition_name]
            service_resource_type = AVAILABLE_ALGORITHMS[service_definition['algorithm']]['resource_type']
            workers = service_definition['workers']
            use_gpu = service_definition['use_gpu']

            service_parameters = {'algorithm': AVAILABLE_ALGORITHMS[service_definition['algorithm']]['prototype'],
                                  'use_gpu': use_gpu}

            if workers is not None:
                service_parameters['pool_limit'] = workers

            service = SERVICE_PROTOTYPE_BY_RESOURCE_TYPE[service_resource_type](**service_parameters)

            self.available_services[service_definition_name] = service

            # We also start the service.
            service.start()

    def face_detection_controller(self):
        """
        Singleton-creation of the face detection controller.
        When this method is invoked, it will add the controller for the face detection to the flask app.
        If the controller already exists, it will only return a reference to it.
        :return: the controller that handles the face detection requests.
        """
        controller_name = 'face_detection_controller'
        services_type = "DETECTION"
        services_subtype = "FACE"

        if controller_name not in self.controllers:
            self.controllers[controller_name] = self._create_atomic_controller(controller_name, services_type,
                                                                               services_subtype)

        return self.controllers[controller_name]

    def age_estimation_controller(self):
        """
        Singleton-creation of the age estimation controller.
        When this method is invoked, it will add the controller for the age estimation to the flask app.
        If the controller already exists, it will only return a reference to it.
        :return: the controller that handles the face detection requests.
        """
        controller_name = "age_estimation_controller"
        services_type = "ESTIMATION"
        services_subtype = "AGE"

        if controller_name not in self.controllers:
            self.controllers[controller_name] = self._create_atomic_controller(controller_name, services_type,
                                                                               services_subtype)

        return self.controllers[controller_name]

    def gender_estimation_controller(self):
        """
        Singleton-creation of the gender estimation controller.
        When this method is invoked, it will add the controller for the gender estimation to the flask app.
        If the controller already exists, it will only return a reference to it.
        :return: the controller that handles the face detection requests.
        """
        controller_name = "gender_estimation_controller"
        services_type = "ESTIMATION"
        services_subtype = "GENDER"

        if controller_name not in self.controllers:
            self.controllers[controller_name] = self._create_atomic_controller(controller_name, services_type,
                                                                               services_subtype)

        return self.controllers[controller_name]

    def _create_atomic_controller(self, controller_name, services_type, services_subtype):
        """
        Generates a controller for the given name (available controllers' names at global CONTROLLERS_LIST var.

        :param controller_name: The name of the controller to create
        :param services_type: Type of the services to inject to the controller.
        :param services_subtype: Subtype of the services to inject to the controller.
        :return: The created controller instance. If the name of the controller does not exist, it will raise an
        exception.
        """

        # The available services for this controller should be a reduced set of services.
        # Reason: it makes no sense to have a feet detection controller with face algorithms.

        filtered_available_services = {}

        if controller_name not in CONTROLLERS_LIST:
            raise Exception("Controller name does not exist.")

        for service_name in self.available_services:
            algorithm_name = self.config.get_services_definition()[service_name]['algorithm']
            type = AVAILABLE_ALGORITHMS[algorithm_name]['type']
            subtype = AVAILABLE_ALGORITHMS[algorithm_name]['subtype']

            if type == services_type and subtype == services_subtype:
                filtered_available_services[service_name] = self.available_services[service_name]

        return CONTROLLERS_LIST[controller_name](flask_web_app=self.flask_app,
                                                 available_services=filtered_available_services,
                                                 config=self.config)

    def face_ensemble_controller(self):
        """
        Singleton-creation of the face ensemble controller.
        When this method is invoked, it will add the controller to the flask app.
        If the controller already exists, it will only return a reference to it.
        :return: the controller that handles ensemble requests as a single request.
        """

        controller_name = 'face_ensemble_controller'

        supported_controllers = [
            'face_detection_controller',
            'age_estimation_controller',
            'gender_estimation_controller'
        ]

        if controller_name not in CONTROLLERS_LIST:
            raise Exception("Controller name does not exist.")

        if controller_name not in self.controllers:

            # This controller does not need a set of available services.
            # Instead, it needs a set of controllers.

            available_controllers = {key: value() for key, value in self.controllers_creation_method.items()
                                     if key in supported_controllers}

            self.controllers[controller_name] = FaceEnsembleController(flask_web_app=self.flask_app,
                                                                       controllers_dict=available_controllers,
                                                                       config=self.config)

        return self.controllers[controller_name]

    def release_all(self, wait_for_release=True):
        """
        Releases all the services and controllers from the APP.
        """

        for controller_name, controller in self.controllers.items():
            controller.release_services(wait_for_release)

        # All the services should be released by this point, however it may have
        # more services injected not passed to a controller.
        for service_name, service in self.available_services.items():
            service.stop(wait_for_release)

    def __del__(self):
        """
        On destruction of the factory, the controllers and services will be gone.
        """

        self.release_all()