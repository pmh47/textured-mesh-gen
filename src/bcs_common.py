
import os
import json
import numpy as np
import tensorflow as tf

import camera_calibration


bcs_raw_path = '../data/2016-ITS-BrnoCompSpeed'
bcs_preprocessed_path = '../preprocessed-data/BrnoCompSpeed/2016-ITS-BrnoCompSpeed'

raw_width, raw_height = 1920, 1080

crop_width, crop_height = 192, 192
crops_per_file = 256

default_camera_height = 8.5  # notional (arbitrary) position of the camera, in metres above the origin


def load_calibration_matrices(dataset_name):

    json_filename = bcs_raw_path + '/results/' + dataset_name + '/system_dubska_optimal_calib.json'
    with open(json_filename) as f:
        results_json = json.load(f)
    first_vp = np.float32(results_json['camera_calibration']['vp1'])
    second_vp = np.float32(results_json['camera_calibration']['vp2'])

    K, R = camera_calibration.get_camera_matrices_from_vanishing_points(
        first_vp, second_vp,
        np.float32([raw_width, raw_height])
    )
    projection_matrix, view_rotation_matrix = camera_calibration.convert_camera_to_gl(
        K, R,
        None,  # so x and y are left in pixel units, and we do our own scaling to ndc after the crop-offset
        near=1., far=100.
    )

    return projection_matrix.astype(np.float32), view_rotation_matrix.astype(np.float32)

