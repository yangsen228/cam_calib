'''
# ATTENTION #
提前准备好内参标定视频放在folder目录下；
每个相机单独一个视频，按照0_in.mp4, 1_in.mp4依次命名；
每个视频仅包含内参棋盘格内容，请提前剪好！
'''

import os
import cv2
import numpy as np
from calib_utils import CamCalibration

cam_num = 3           # 相机个数
folder = '20200814'   # 改成标定当天日期
if not os.path.exists(folder):
    os.makedirs(folder)

# 内参标定
M, dists = [], []
for cam_idx in range(cam_num):
    print('========= calibrate cam{} ========='.format(cam_idx))
    video_path = '{}/{}_in.mp4'.format(folder, cam_idx)

    my_calib = CamCalibration(28.71, 8, 6)                                       # 如更换棋盘格请修改：(棋盘格边长(mm)，横向内角点个数，纵向内角点个数)
    cameraMatrix, distCoeffs = my_calib.inCalibration(video_path, check=True)
    print(cameraMatrix)
    print(distCoeffs)

    M.append(cameraMatrix)
    dists.append(distCoeffs)
M, dists = np.array(M), np.array(dists)
print(M.shape, dists.shape)

# 保存内参
np.save('{}/cameraMatrix.npy'.format(folder), M)
np.save('{}/distCoeffs.npy'.format(folder), dists)