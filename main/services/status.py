#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Ivan de Paz Centeno"


# **********************************
# SERVICE GENERIC STATUS
# **********************************
# You can define more service status flags:
SERVICE_RUNNING = 1
# ----
# All the flags above 1 are considered as service running.
# Example:
# SERVICE_FETCHING_DATA = 1

SERVICE_STOPPED = 0
# ----
# All the flags below 0 are considered as service stopped.
# Example:
# SERVICE_CRASHED = -2


# **********************************
# SERVICE DATASET BUILDER STATUS
# **********************************
SERVICE_STOPPING = -1
SERVICE_STATUS_UNKNOWN = 999999

CODE_MAP = {
    SERVICE_RUNNING: "SERVICE_RUNNING",
    SERVICE_STOPPED: "SERVICE_STOPPED",

    SERVICE_STOPPING: "SERVICE_STOPPING",

    SERVICE_STATUS_UNKNOWN: "SERVICE_STATUS_UNKNOWN",
}

CODE_MAP_INV = {v: k for k, v in CODE_MAP.items()}


def status_code_to_name(status_code):
    """
    Converts the status code to name
    :param status_code:
    :return:
    """
    if status_code in CODE_MAP:
        result = CODE_MAP[status_code]
    else:
        result = "SERVICE_STATUS_UNKNOWN"

    return result


def name_to_status_code(status_code_name):
    """
    Converts the name into a status code.
    :param status_code_name: example of status code name: "SERVICE_STOPPED"
    :return: status code for the name specified.
    """

    if status_code_name in CODE_MAP_INV:
        result = CODE_MAP_INV[status_code_name]
    else:
        result = SERVICE_STATUS_UNKNOWN

    return result
