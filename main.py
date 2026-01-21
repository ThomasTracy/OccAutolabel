from cores.config.config import Config
import numpy as np
import yaml
import os
import sys

# from occ_autolabel import OCCAutolabelwithMask, OCCAutolabelwithLidarSegmentation
from cores.autolabel import OCCAutolabelwithLidarSegmentation, OCCAutolabelwithUEPose


if __name__ == '__main__':

    config_path = "/home/jszr/Code/Autolabel/occ_autolabel/configs/config.yaml"
    autolabel = OCCAutolabelwithUEPose(config_path)
    # autolabel = OCCAutolabelwithLidarSegmentation(config_path)
    # semantic_points = autolabel.process_single_frame('1761112164_540160000')
    autolabel.process_multi_frame()
