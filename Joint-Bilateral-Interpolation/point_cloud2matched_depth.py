import glob
import os
import cv2
import numpy as np
import json
from open3d import *
import PIL
from PIL import Image
import pandas as pd


def _get_pc_flat_using_all_coeff(pcflat_rgb, fx, fy, cx, cy, k1, k2, k3, p1, p2):
    rgb_camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0], ])
    rgb_dist_coeffs = np.array([[k1, k2, p1, p2, k3]])
    pcflat_rgb_, _ = cv2.projectPoints(pcflat_rgb.reshape(-1, 3), (0, 0, 0), (0, 0, 0), rgb_camera_matrix,
                                       rgb_dist_coeffs)

    pcflat_rgb2 = pcflat_rgb.copy()
    pcflat_rgb2[:, 0:2] = pcflat_rgb_.reshape((-1, 2))

    return pcflat_rgb2


def point_cloud_in_rgb(pc, H_tof_rgb, fx, fy, cx, cy,k1, k2, k3, p1, p2, use_KP_coeff=False, invalid=32001):
    pc = pc.reshape((-1, 3))
    valid = pc[:, 2] < invalid

    pcflat_h = np.ones((pc.shape[0], 4))
    pcflat_h[:, 0:3] = pc[:, 0:3]

    pcflat_rgb_h = np.dot(pcflat_h, H_tof_rgb.T)
    pcflat_rgb = pcflat_rgb_h[:, 0:3]
    pcflat_rgb = pcflat_rgb[valid]
    pcflat_rgb_kp = _get_pc_flat_using_all_coeff(pcflat_rgb, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    return pcflat_rgb_kp


def _round_positive_values(data):
    return np.floor(data + 0.5).astype(np.int16)  # rounding


def _apply_pc_in_image(pc, img_shape):
    valid = ((0 <= pc[:, 0]) & (pc[:, 0] < img_shape[1])) & ((0 <= pc[:, 1]) & (pc[:, 1] < img_shape[0]))

    coord = pc[valid]

    depth_aligned = np.ones(img_shape, dtype=np.int16) * 32001

    depth_aligned[coord[:, 1], coord[:, 0]] = (coord[:, 2])

    return depth_aligned


def get_depth_aligned_from_projected_point_cloud(pcflat_rgb, rgb_shape, new_shape=None):
    if new_shape is None:
        new_shape = rgb_shape
    ratio = (new_shape[1] / rgb_shape[1], new_shape[0] / rgb_shape[0], 1.0)
    pcflat_rgb *= ratio
    pc = _round_positive_values(pcflat_rgb)  # rounding
    depth_aligned = _apply_pc_in_image(pc, new_shape)
    return depth_aligned


if __name__ == '__main__':
    # float_z.shape (60, 120, 160)
    # float_pc (115200, 3)
    # rgb (60, 616, 820, 3)
    calib_file = open("/Users/Vision/01_Project/02_STC-SONY/99_Data/RGBD_fusion_records_QianJian_npy"
                      "/calibration_BKF518_00_P145.json")
    calib_parameter = json.load(calib_file)
    intrinsics = calib_parameter['rgb2_intrinsics'][0]['intrinsics']
    rmat3x3 = np.array(calib_parameter['extrinsics']['tof_to_rgb2']['rmat3x3'])
    tvec = np.array(calib_parameter['extrinsics']['tof_to_rgb2']['tvec'])
    H_tof_rgb = np.hstack([rmat3x3.reshape(3, 3), tvec.reshape(3, 1)])
    I = [0, 0, 0, 1]
    H_tof_rgb_I = np.vstack([H_tof_rgb, I])

    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    k1 = intrinsics.get("k1")
    k2 = intrinsics.get("k2")
    k3 = intrinsics.get("k3")
    p1 = intrinsics.get("p1")
    p2 = intrinsics.get("p2")
    for x in range(0, 1):
        pc_datapath = "/Users/Vision/01_Project/02_STC-SONY/03_RGBD-Fusion/20210629_SpotToF/BKF_PointCloud/depth_org_" \
                      "%d.npy" % x
        pc_dataset = np.load(pc_datapath)
        pcflat_rgb_kp = point_cloud_in_rgb(pc_dataset, H_tof_rgb_I, fx, fy, cx, cy, k1, k2, k3, p1, p2, use_KP_coeff=False,
                                           invalid=32001)
        print(pcflat_rgb_kp.shape)
        rgb_shape = [616, 820]
        new_shape = [616, 820]
        depthmap = get_depth_aligned_from_projected_point_cloud(pcflat_rgb_kp, rgb_shape, new_shape)
        for i in range(0, 616):
            for j in range(0, 820):
                if depthmap[i, j] == 32001:
                    depthmap[i, j] = 0
                elif depthmap[i, j] < 5:
                    depthmap[i, j] = 0
        print("depthmap=", depthmap)
        np.save("BKF_Matched_Depth-304-228/depth_%d.npy" % x, depthmap)
        cv2.imwrite("BKF_Matched_Depth-304-228/depth_%d.jpg" % x, depthmap)
        data = pd.DataFrame(depthmap)
        data.to_csv('BKF_Matched_Depth-304-228/depth_%d.csv' % x, index=False, header=False)
