#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from main.controllers.controller import route
from main.controllers.image_controller import ImageController
from main.exceptions.invalid_request import InvalidRequest

__author__ = "Ivan de Paz Centeno"


class FaceEnsembleController(ImageController):
    """
    Controller for /ensemble-requests/faces/ URL
    This allows to execute piped requests to multiple controllers and serialize it through the same controller.
    For example, it allows to perform a detection of faces and pipe each of them to the estimation algorithms.
    """

    def __init__(self, flask_web_app, controllers_dict, config):
        """
        Constructor of the Face Ensemble controller.
        :param flask_web_app: web app from Flask already initialized.
        :param controllers_dict: a dictionary with the controllers
        :param config: config object containing all the service definitions.
        """
        ImageController.__init__(self, flask_web_app, {}, config, "ENSEMBLE", "FACE")
        self.controllers_dict = controllers_dict

        self.exposed_methods += [
            self.detect_face_estimate_age_gender_from_base64,
            self.detect_face_estimate_age_gender_from_stream,
            self.get_available_services
        ]

        self._init_exposed_methods()

    def _get_most_suitable_service(self, service_name, controller=None):
        """
        Searchs for the most suitable service for the requested service name.
        If a service name is provided but no services matches it, it will return an InvalidRequest exception.
        :param service_name: name of the service to match. If it is 'default', it will fall back to the
                             default service for this controller.
        :param controller: controller instance to retrieve from.
        :return: the most suitable service for the service name.
        """

        if service_name == "none" or controller is None:
            service = None

        else:
            service = ImageController._get_most_suitable_service(controller, service_name)

        return service

    @route("/ensemble-requests/faces/services", methods=['GET'])
    def get_available_services(self):
        """
        Retrieves the services available for ensemble per controller.
        """

        available_services = {}

        for controller_name, controller in self.controllers_dict.items():
            available_services[controller_name] = ImageController.get_available_services(controller)

        return jsonify(available_services)

    @route("/ensemble-requests/faces/detection-estimation-age-gender/base64", methods=['PUT'])
    def detect_face_estimate_age_gender_from_base64(self):
        """
        Performs a detection of in a base64 stream. It will pipe the detections bounding
        boxes to the given age and gender estimation service

        The requests accepts the following parameters:
          [OPTIONAL]    service_face=SERVICE_NAME           # Service for face detection.
          [OPTIONAL]    service_age=SERVICE_NAME            # Service for age estimation.
          [OPTIONAL]    service_gender=SERVICE_NAME         # Service for gender estimation.
          [OPTIONAL]    limit_estimations=LIMIT_NUM         # Number of faces that disable the estimation if reached.
                                                              (default: 3)
          [OPTIONAL]    bounding_box_expansion=PROPORTION   # Proportion in float of expansion of the bbox(default: 0.8)
          [OPTIONAL]    work_in_gray=true/false             # Works in grayscale or not. (default: true)

        :return: The detection result in JSON format (bounding boxes). Example of result:
          [
              "0" : {
                "height": 74,
                "width": 75,
                "x": 460,
                "y": 179,
                "age": [2,4]
                "gender": "Female"
              },
              "1" : {
                "height": 44,
                "width": 67,
                "x": 260,
                "y": 150,
                "age": [12,24]
                "gender": "Male"
              }
          ]

        """

        request, service_face_detection, service_age_estimation, \
                 service_gender_estimation, work_in_gray, limit_estimations, \
                 bounding_box_expansion = self._get_common__parameters()

        content = self._get_raw_content_validated(is_base64=True)
        image = self._build_image_from_content(content, work_in_gray)

        return jsonify(self._process_face_age_gender_image(image, service_face_detection, service_age_estimation,
                                                           service_gender_estimation,
                                                           bounding_box_expansion, limit_estimations))

    @route("/ensemble-requests/faces/detection-estimation-age-gender/stream", methods=['PUT'])
    def detect_face_estimate_age_gender_from_stream(self):
        """
        Performs a detection of in a binary stream. It will pipe the detections bounding
        boxes to the given age and gender estimation service

        The requests accepts the following parameters:
          [OPTIONAL]    service_face=SERVICE_NAME           # Service for face detection.
          [OPTIONAL]    service_age=SERVICE_NAME            # Service for age estimation.
          [OPTIONAL]    service_gender=SERVICE_NAME         # Service for gender estimation.
          [OPTIONAL]    limit_estimations=LIMIT_NUM         # Number of faces that disable the estimation if reached.
                                                              (default: 3)
          [OPTIONAL]    bounding_box_expansion=PROPORTION   # Proportion in float of expansion of the bbox(default: 0.8)
          [OPTIONAL]    work_in_gray=true/false             # Works in grayscale or not. (default: true)

        :return: The detection result in JSON format (bounding boxes). Example of result:
          [
              "0" : {
                "height": 74,
                "width": 75,
                "x": 460,
                "y": 179,
                "age": [2,4]
                "gender": "Female"
              },
              "1" : {
                "height": 44,
                "width": 67,
                "x": 260,
                "y": 150,
                "age": [12,24]
                "gender": "Male"
              }
          ]

        """

        request, service_face_detection, service_age_estimation, \
                 service_gender_estimation, work_in_gray, limit_estimations, \
                 bounding_box_expansion = self._get_common__parameters()

        content = self._get_raw_content_validated(is_base64=False)
        image = self._build_image_from_content(content, work_in_gray)

        return jsonify(self._process_face_age_gender_image(image, service_face_detection, service_age_estimation,
                                                           service_gender_estimation,
                                                           bounding_box_expansion, limit_estimations))

    def _process_face_age_gender_image(self, image, face_service, age_service, gender_service,
                                       bounding_box_expansion, limit_estimations):
        """
        Automates the process of calculating the parameters for the faces when the parameters have been retrieved from
        the request. All the requests for face + age + gender share this behaviour.
        :param image: image to process
        :param face_service: service for the detection of faces
        :param age_service: service for the estimation of ages
        :param gender_service: service for the estimation of genders
        :param bounding_box_expansion: expansion of the bounding box to pipe to the estimation services
        :param limit_estimations: number of boundingboxes that disable the estimation pipeline for increasing
        performance.
        :return: result as json.
        """
        face_detection_result = face_service.append_request(image).get_resource()

        bounding_boxes = self._retrieve_result_metadata(face_detection_result)

        # let's expand the bounding boxes:
        for bounding_box in bounding_boxes:
            bounding_box.expand(bounding_box_expansion)
            bounding_box.fit_in_size(image.get_size())

        result_set = None
        cached_crops = {}

        result_set, \
        cached_crops = self._build_result_set_promises_from_bounding_boxes(image, bounding_boxes,
                                                                           age_service,
                                                                           promise_identity="age",
                                                                           previous_result_promises=result_set,
                                                                           cached_crops=cached_crops,
                                                                           limit_estimations=limit_estimations)

        result_set, \
        cached_crops = self._build_result_set_promises_from_bounding_boxes(image, bounding_boxes,
                                                                           gender_service,
                                                                           promise_identity="gender",
                                                                           previous_result_promises=result_set,
                                                                           cached_crops=cached_crops,
                                                                           limit_estimations=limit_estimations)

        return self._fetch_results_as_json(result_set)

    @staticmethod
    def _build_result_set_promises_from_bounding_boxes(image, bounding_boxes, service_to_get_promise_from,
                                                       promise_identity="default",
                                                       previous_result_promises=None, cached_crops=None,
                                                       limit_estimations=3):
        """
        Constructs a result set for bounding boxes that may have attached more services.
        :param image: full image to process. If cached_crops is filled, this attribute can be ignored (None).
        :param bounding_boxes: list of bounding-boxes objects.
        :param service_to_get_promise_from: service to append a request of the cropped-by-bbox image.
        :param promise_identity: name to identify the promise in the result. By default it will "default". Ensure
                                 to give a unique name in the case this result is piped to this method again in order
                                 to apply a new service request.
        :param previous_result_promises: the result of a previous build. This is helpful to pipe multiple builds
                                         of different services into the same result. This means that the result is
                                         going to be this previous result but extended with some more information.
        :param cached_crops: the cached-crops images. If this attribute is filled, this method will use it instead
                             of working with the full image. If not, it will create it lazily.
        :param limit_estimations: number of bounding boxes that, when overpassed, will disable estimation algorithms.
                             Set to 0 to disable this behaviour.
        :return: the result set filled with promise objects from the service and the cached crops.
        """

        result = previous_result_promises

        if result is None:
            result = {}

        if cached_crops is None:
            cached_crops = {}

        index = -1

        for bbox in bounding_boxes:

            index += 1

            # We need to associate the index with the promise result of each request.
            # Reason: a face must have a reference to its age and gender.
            if index not in result:
                result[index] = {'bounding_box': bbox}

            if len(bounding_boxes) > limit_estimations != 0:
                continue

            if index in cached_crops:
                cropped_image = cached_crops[index]
            else:
                # We only crop the image when there's a need (lazy crop).
                cropped_image = None

            if service_to_get_promise_from is None:
                continue

            if cropped_image is None:
                cropped_image = image.crop_image(bbox, 'face id {}'.format(index))
                cached_crops[index] = cropped_image

            result_promise = service_to_get_promise_from.append_request(cropped_image)

            # We don't fetch the resource from the promise until we already have appended all the requests.
            # This allows us to process them in parallel.
            result[index][promise_identity] = result_promise


        return result, cached_crops

    def _fetch_results_as_json(self, result_set):
        """
        Fetchs the results of the estimation algorithms.
        :param result_set: result set of face indexes associated with bounding boxes objects and age and gender promises.
        :return: Pure-JSON result of the process.
        """
        result_json = {}

        # Now we need to fetch the data from the promises and fill a JSON response with the result.
        for index, face in result_set.items():
            result_json[index] = {
                'Face_ID': index,
                'Bounding_box': face['bounding_box'].__str__()
            }

            if 'age' in face:
                age_result = face['age'].get_resource()
                age_range = self._retrieve_result_metadata(age_result)[0]
                result_json[index].update(age_range.to_dict())

            if 'gender' in face:
                gender_result = face['gender'].get_resource()
                gender = self._retrieve_result_metadata(gender_result)[0]
                result_json[index].update(gender.to_dict())

                # Here it can be appended more (if 'blabla' in face:) to be widely usable in the
                # whole controller, instead of only age/gender estimation.

        return result_json

    @staticmethod
    def _validate_number(var_name, number):
        """
        Checks whether the specified parameter is a number or not. If not, it will raise a exception.
        :param var_name: Name of the var to show in the exception (in order to reference the error).
        :param number: Number to validate.
        """

        try:
            num = int(number)  # for int, long and float

            if num < 0:
                raise ValueError("Negative integers not allowed.")

        except ValueError:
            raise InvalidRequest("The parameter {} is not a valid natural integer.".format(var_name))

    @staticmethod
    def _validate_float(var_name, float_number):
        """
        Checks whether the specified parameter is a number or not. If not, it will raise a exception.
        :param var_name: Name of the var to show in the exception (in order to reference the error).
        :param number: Number to validate.
        """

        try:
            float(float_number)  # for int, long and float
        except ValueError:
            raise InvalidRequest("The parameter {} is not a valid float.".format(var_name))

    def _get_common__parameters(self, should_have_uri=False):
        """
        Retrieves the common parameters for face ensemble requests.
        :param should_have_uri: if the flag is True, the validation of the request will check that an URI is
        present inside the request. This is a flag forwarded to _get_validated_request() method.
        :return: request, service_face_detection, service_age_estimation,
                 service_gender_estimation, work_in_gray, limit_estimations,
                 bounding_box_expansion
        """
        face_controller = age_controller = gender_controller = None

        request = FaceEnsembleController._get_validated_request(should_have_uri=should_have_uri)

        limit_estimations = int(request.get('limit_estimations', 3))
        bounding_box_expansion = float(request.get('bounding_box_expansion', 0.8))
        work_in_gray = request.get('work_in_gray', "true") == "true"

        self._validate_number("limit_estimations", limit_estimations)
        self._validate_float("bounding_box_expansion", bounding_box_expansion)

        service_face_name = request.get('service_face', 'default')
        service_age_name = request.get('service_age', 'default')
        service_gender_name = request.get('service_gender', 'default')

        if 'face_detection_controller' in self.controllers_dict:
            face_controller = self.controllers_dict['face_detection_controller']
        if 'age_estimation_controller' in self.controllers_dict:
            age_controller = self.controllers_dict['age_estimation_controller']
        if 'gender_estimation_controller' in self.controllers_dict:
            gender_controller = self.controllers_dict['gender_estimation_controller']

        service_face_detection = self._get_most_suitable_service(service_face_name, face_controller)
        service_age_estimation = self._get_most_suitable_service(service_age_name, age_controller)
        service_gender_estimation = self._get_most_suitable_service(service_gender_name, gender_controller)

        if not service_face_detection:
            raise InvalidRequest("No services for face detection found.")

        return request, service_face_detection, service_age_estimation, \
               service_gender_estimation, work_in_gray, limit_estimations, \
               bounding_box_expansion
