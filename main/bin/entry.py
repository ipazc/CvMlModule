#!/usr/bin/env python
# -*- coding: utf-8 -*-

from main.controllers.controller_factory import ControllerFactory
from main.model.config import Config, fix_working_dir
from flask import Flask

# We need to import those algorithms and services that we want to have in our app since this will trigger their
# registration.
import main.model.algorithm.detection.face.opencv_haar_cascade_face_detection_algorithm
import main.model.algorithm.detection.face.dlib_hog_svm_face_detection_algorithm
import main.model.algorithm.detection.face.mt_cnn_face_detection_algorithm
import main.model.algorithm.estimation.age.levi_hassner_cnn_age_estimation_algorithm
import main.model.algorithm.estimation.gender.levi_hassner_cnn_gender_estimation_algorithm
import main.services.image.algorithm_service


__author__ = "Ivan de Paz Centeno"

app = Flask(__name__)

fix_working_dir()

config = Config()

web_app_definition = config.get_webapp_definition()

controller_factory = ControllerFactory(app, config)

# Now we set up which controllers do we want to hold in our APP.
controller_factory.face_detection_controller()
controller_factory.age_estimation_controller()
controller_factory.gender_estimation_controller()
controller_factory.face_ensemble_controller()

app.run(web_app_definition['ip'], web_app_definition['port'], threaded=True)

controller_factory.release_all()
