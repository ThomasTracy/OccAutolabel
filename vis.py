import cv2
import numpy as np
from cores.utils.visualize import visualize_voxels, visualize_pc_on_image

if __name__ == "__main__":
    # colors = np.array(
    #     [
    #         [128, 64, 128],
    #         [244, 35, 232],
    #         [70, 70, 70],
    #         [102, 102, 156],
    #         [190, 153, 153],
    #         [153, 153, 153],
    #         [250, 170, 30],
    #         [220, 220, 0],
    #         [107, 142, 35],
    #         [152, 251, 152],
    #         [70, 130, 180],
    #         [220, 20, 60],
    #         [255, 0, 0],
    #         [0, 0, 142],
    #         [0, 0, 70],
    #         [0, 60, 100],
    #         [0, 80, 100],
    #         [0, 0, 230],
    #         [119, 11, 32]
    #     ]
    # ).astype(np.uint8)
    # voxel_size = 0.5
    # voxel_file = '/home/jszr/Data/OCC_Autolabel/output/1532402927647951.npy'
    # voxel_data = np.load(voxel_file)    
    # visualize_voxels(voxel_data, colors, voxel_size)

    image = cv2.imread("/home/jszr/Data/OCC_Autolabel/data/camera/CAM_BACK_LEFT/1532402927647951.jpg",cv2.COLOR_RGB2BGR)
    points = np.load("debug.npy")
    visualize_pc_on_image(image, points)