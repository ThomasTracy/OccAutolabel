from cores.config.config import Config
import numpy as np
import yaml
import os

from occ_autolabel import OCCAutolabel


if __name__ == '__main__':
    # postprocessor = Postprocessor()
    # data = np.random.rand(10, 10)
    # postprocessor.process(data)
    config_path = "/home/jszr/Code/Autolabel/occ_autolabel/configs/config.yaml"
    data_root = "/home/jszr/Data/OCC_Autolabel/data"
    save_path = "/home/jszr/Data/OCC_Autolabel/output"
    label_map = "/home/jszr/Code/Autolabel/occ_autolabel/configs/label_mapping.yaml"

    config = Config(config_path)
    with open(label_map, 'r') as f:
        label_mapping = yaml.safe_load(f)
    data_dirs = {
        'save_path': save_path,
        'calibration': os.path.join(data_root, 'calibration','calibration.json'),
        'camera':  os.path.join(data_root, 'camera'),
        'lidar': os.path.join(data_root, 'lidar'),
        'mask': os.path.join(data_root, 'masks'),
        'pose': os.path.join(data_root, 'pose', 'pose.json'),
        'label_mapping': label_map
    }
    autolabel = OCCAutolabel(data_dirs, config)
    semantic_points = autolabel.process_single_frame('1532402927647951')