import numpy as np

def euler_to_rotation(rx,ry,rz):
    # 通过旋转角计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    return R

def compute_extrinsic(pos, angular):
    tx, ty, tz = pos
    rx, ry, rz = angular
    R = euler_to_rotation(rx, ry, rz)
    T = -R @ np.array([tx, ty, tz])
    extrinsic_matrix = np.vstack((R, T))
    return extrinsic_matrix, R, T

if __name__ == '__main__':

    extrinsic, R, T = compute_extrinsic([0.0, -0.12, 0.0], [0.0, 0.0, np.pi/2])
    print("----------------------------")
    print("R: ",R)
    print("T: ",T)
    print("extrinsic: ", extrinsic)

    # rot = np.array([5.1039238286220137e+02, 0., 9.5744966457609075e+02, 0.,
    #    5.1000106359433551e+02, 7.7814230197535107e+02, 0., 0., 1.]).reshape(3, 3)
    # rot = np.round(rot, 2)
    # trans = np.array([4.7385600198694389e-01, 6.3216972842159033e-02,
    #    5.1177406699536687e-01])
    # trans = np.round(trans, 2)
    # print("rotation matrix: \n", rot)
    # print("translation matrix: \n", trans)
