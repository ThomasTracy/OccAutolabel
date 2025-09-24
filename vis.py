import cv2
import numpy as np
import open3d as o3d
from cores.utils.visualize import visualize_voxels, visualize_pc_on_image
from cores.utils.projector import Projector

def vis_voxel():
    colors = np.array(
        [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]
        ]
    ).astype(np.uint8)
    voxel_size = 0.5
    voxel_file = '/home/jszr/Data/OCC_Autolabel/output/1532402929197353.npy'
    voxel_data = np.load(voxel_file)    
    visualize_voxels(voxel_data, colors, voxel_size)

def vis_pc_on_image():
    image = cv2.imread("/home/jszr/Data/OCC_Autolabel/data/camera/CAM_FRONT/1532402929197353.jpg",cv2.COLOR_RGB2BGR)
    points = np.load("murged_points.npy")
    # points = np.fromfile("/home/jszr/Data/OCC_Autolabel/data/lidar/1532402929197353.bin", dtype=np.float32).reshape(-1, 4)
    print(points.shape)
    projector = Projector()
    image_shape = (1600,900)
    camera_matrix = np.array([
                [
                    1266.417203046554,
                    0.0,
                    816.2670197447984
                ],
                [
                    0.0,
                    1266.417203046554,
                    491.50706579294757
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ])
    distortion_coeffs = None
    camera_to_ego_rotation = np.array([
                    0.4998015430569128,
                    -0.5030316162024876,
                    0.4997798114386805,
                    -0.49737083824542755
                ])
    camera_to_ego_translation = np.array([
                    1.70079118954,
                    0.0159456324149,
                    1.51095763913
                ])
    # lidar_to_ego_rotation = np.array([
    #                 0.7077955119163518,
    #                 -0.006492242056004365,
    #                 0.010646214713995808,
    #                 -0.7063073142877817
    #             ])
    # lidar_to_ego_translation = np.array([
    #                             0.943713,
    #                             0.0,
    #                             1.84023
    #                         ])
    lidar_to_ego_rotation = None
    lidar_to_ego_translation = None
    points2d, valid_mask = projector.projection(points,
                                                    image_shape=image_shape,
                                                    camera_matrix=camera_matrix,
                                                    distortion_coeffs=distortion_coeffs,
                                                    camera_to_ego_rotation=camera_to_ego_rotation,
                                                    camera_to_ego_translation=camera_to_ego_translation,
                                                    lidar_to_ego_rotation=lidar_to_ego_rotation,
                                                    lidar_to_ego_translation=lidar_to_ego_translation
                                                    )
    visualize_pc_on_image(image, points2d)

def visualize_points():
    points = np.load("/home/jszr/Code/Autolabel/occ_autolabel/full_points.npy")
    colors=None
    point_size=2.0
    window_name="3D Point Cloud"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 如果提供了颜色信息
    if colors is not None:
        # 确保颜色值在[0, 1]范围内
        if np.max(colors) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 默认设置为灰色
        pcd.colors = o3d.utility.Vector3dVector(np.full((points.shape[0], 3), [0.5, 0.5, 0.5]))

    # 可视化
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # vis_pc_on_image()
    visualize_points()
    # vis_voxel()