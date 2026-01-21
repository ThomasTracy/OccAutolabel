import os 
import json


if __name__ == "__main__":
    camera_front = {
        "rotation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "translation": [
            -0.29,
            0.0,
            -0.01
        ],
        "distortion_coeffs": [1.2989676453174887e-01, -2.2790603406357010e-02,
                            -1.1560361547857329e-02, 3.6854238155112160e-03],
        "intrinsic": [
            960.0, 0., 960.0,
            0., 960.0, 768.0, 
            0., 0., 1. 
            ]
    }
    camera_back = {
        "rotation": [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "translation": [
            -0.29,
            0.0,
            -0.01
        ],
        "distortion_coeffs": [1.2553636657753328e-01, -1.6262291205407977e-02,
                        -1.5083821029898107e-02, 4.4097186717246689e-03 ],
        "intrinsic": [
            960.0, 0., 960.0,
            0., 960.0, 768.0, 
            0., 0., 1. 
            ]
    }
    camera_left = {
        "rotation": [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "translation": [
            0.12,
            0.0,
            0.0
        ],
        "distortion_coeffs": [1.3312982315133456e-01, -3.0421672886919042e-02,
                            -5.1777009986598496e-03, 1.8360985440472463e-03],
        "intrinsic": [
            960.0, 0., 960.0,
            0., 960.0, 768.0, 
            0., 0., 1. 
            ]
    }
    camera_right = {
        "rotation": [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "translation": [
            0.12,
            0.0,
            0.0
        ],
        "distortion_coeffs": [1.3452624228830984e-01, -3.4332439405819079e-02,
                            -3.8548614050705914e-03, 1.8646739133350483e-03 ],
        "intrinsic": [
            960.0, 0., 960.0,
            0., 960.0, 540.0, 
            0., 0., 1. 
            ]
    }
    lidar = {
        "rotation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "translation": [
            0.0,
            0.0,
            0.0
        ]
    }

    json_data = {
        "lidar": lidar,
        "camera_front": camera_front,
        "camera_back": camera_back,
        "camera_left": camera_left,
        "camera_right": camera_right
    }

    save_path = "/home/jszr/Data/OCC_Autolabel/DATA/test_beida/calibration.json"
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)
