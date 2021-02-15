'''
# ATTENTION #
提前准备好外参标定图片放在folder目录下；
每个相机单独一张图片，按照0_ex.jpg, 1_ex.jpg依次命名；
这些图片必须经过严格时钟同步!
'''

import os
import cv2
import pickle
import numpy as np
from calib_utils import CamCalibration

cam_num = 3           # 相机个数
folder = '20200814'   # 改成标定当天日期
if not os.path.exists(folder):
    os.makedirs(folder)

# 载入内参
M = np.load('{}/cameraMatrix.npy'.format(folder))
dists = np.load('{}/distCoeffs.npy'.format(folder))

# 外参标定
my_calib = CamCalibration(145, 8, 6)     # 如更换棋盘格请修改：(棋盘格边长(mm)，横向内角点个数，纵向内角点个数)
cam_params = {'K':[],'P':[],'RT':[],'r':[],'t':[]}
for cam_idx in range(cam_num):
    print('========= calibrate cam{} ========='.format(cam_idx))
    cam_params['K'].append(M[cam_idx])
    print(M[cam_idx])

    img = cv2.imread('{}/{}_ex.jpg'.format(folder, cam_idx))
    R, rvec, tvec = my_calib.exCalibration(img, M[cam_idx], dists[cam_idx])
    cam_params['r'].append(rvec)
    cam_params['t'].append(tvec)
    print(rvec, tvec)
    RT = np.concatenate((R, tvec / 1000.0), axis=1)
    cam_params['RT'].append(RT)
    print(RT)
    P = M[cam_idx] @ RT
    cam_params['P'].append(P)
    print(P)
cam_params['K'] = np.array(cam_params['K'])
cam_params['P'] = np.array(cam_params['P'])
cam_params['RT'] = np.array(cam_params['RT'])
cam_params['r'] = np.array(cam_params['r'])
cam_params['t'] = np.array(cam_params['t'])

# 按mvpose要求保存内外参
with open('{}/camera_parameter.pickle'.format(folder), 'wb') as f:
    pickle.dump(cam_params, f)