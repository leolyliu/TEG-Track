import os
import numpy as np


def train_test_split():
    training_videonames = [
        "0907_camera2_1",  # 5
        "0907_camera2_2",  # 4
        "0907_bottle1_1",  # 5
        "0907_bottle1_2",  # 5
        "0907_camera2_3",  # 3
        "0907_can1_1",  # 5
        "0907_can1_2",  # 5
        "0907_can1_3",  # 2
        "0907_bottle1_3",  # 4
        "0910_camera4_1",  # 5
        "0910_camera4_4",  # 5
        "0910_can3_1",  # 5
        "0910_can3_4",  # 5
        "0913_bottle3_1",  # 5
        "0913_bottle3_3",  # 5
    ]
    test_videonames = [
        "0907_camera3_1",
        "0907_camera3_2",
        "0907_camera3_3",
        "0910_camera5_1",
        "0910_camera5_4",
        "0907_bottle2_1",
        "0907_bottle2_2",
        "0907_bottle2_3",
        "0910_bottle2_4",
        "0913_bottle4_1",
        "0913_bottle4_5",
        "0907_bowl1_1",
        "0907_bowl1_2",
        "0907_bowl1_3",
        "0910_bowl1_4",
        "0913_bowl2_1",
        "0913_bowl2_3",
        "0913_bowl2_4",
        "0913_bowl2_5",
        "0907_can2_1",
        "0907_can2_2",
        "0907_can2_3",
        "0910_can4_1",
        "0910_can4_4",
        "0907_mug2_1",
        "0907_mug2_2",
        "0907_mug2_3",
        "0907_mug3_1",
        "0907_mug4_1",
        "0907_mug4_2",
        "0907_mug4_3",
        "0910_mug4_4",
        "0913_mug3_5",
        "0913_mug4_5",
    ]
    return training_videonames, test_videonames


def get_slippage_testset():
    test_vn_range = [
        ["0907_camera3_1", [80, 100]],
        ["0907_camera3_1", [430, 450]],
        ["0907_camera3_1", [855, 875]],
        ["0907_camera3_1", [1238, 1258]],
        ["0907_camera3_1", [1472, 1492]],
        ["0907_camera3_1", [1563, 1583]],
        ["0907_camera3_1", [1850, 1870]],
        ["0907_camera3_3", [140, 160]],
        ["0907_camera3_3", [828, 848]],
        ["0907_camera3_3", [1249, 1269]],
        ["0907_camera3_3", [1638, 1658]],
        ["0907_bottle2_1", [263, 283]],
        ["0907_bottle2_1", [438, 458]],
        ["0907_bottle2_1", [902, 922]],
        ["0907_bottle2_1", [1262, 1282]],
        ["0907_bottle2_1", [1633, 1653]],
        ["0913_bottle4_1", [30, 50]],
        ["0913_bottle4_5", [81, 101]],
        ["0907_can2_1", [202, 222]],
        ["0907_can2_1", [313, 333]],
        ["0907_can2_1", [348, 368]],
        ["0907_can2_1", [373, 393]],
        ["0907_can2_1", [425, 445]],
        ["0907_can2_1", [496, 516]],
        ["0907_can2_1", [571, 591]],
        ["0907_can2_1", [615, 635]],
        ["0907_can2_1", [631, 651]],
        ["0907_can2_1", [743, 763]],
        ["0907_can2_1", [758, 778]],
        ["0907_can2_1", [776, 796]],
        ["0907_can2_3", [1475, 1495]],
        ["0907_can2_3", [1529, 1549]],
        ["0907_can2_3", [1565, 1585]],
        ["0910_can4_1", [860, 880]],
        ["0910_can4_4", [1700, 1720]],
        ["0907_mug2_1", [137, 157]],
        ["0907_mug2_1", [757, 777]],
        ["0907_mug2_1", [849, 869]],
        ["0907_mug2_1", [916, 936]],
        ["0907_mug2_1", [1063, 1083]],
        ["0907_mug2_1", [1111, 1131]],
        ["0907_mug2_1", [1243, 1263]],
        ["0907_mug2_1", [1258, 1278]],
        ["0907_mug2_1", [1579, 1599]],
        ["0907_mug2_1", [1658, 1678]],
        ["0907_mug2_3", [942, 962]],
        ["0907_mug2_3", [1372, 1392]],
        ["0907_mug2_3", [1594, 1614]],
        ["0907_mug2_3", [1872, 1892]],
        ["0907_mug4_1", [255, 275]],
        ["0910_mug4_4", [755, 775]],
        ["0910_mug4_4", [1716, 1736]],
        ["0910_mug4_4", [1804, 1824]],
        ["0910_mug4_4", [1847, 1867]],
        ["0910_mug4_4", [1904, 1924]],
        ["0910_mug4_4", [1949, 1969]],
        ["0913_mug4_5", [186, 206]],
        ["0913_mug4_5", [673, 693]],
    ]
    return test_vn_range


def shape_xyzs():
    shape_xyzs = {
        "camera1": np.array([0.029, 0.055, 0.077]),
        "camera2": np.array([0.050, 0.066, 0.118]),
        "camera3": np.array([0.042, 0.081, 0.110]),
        "camera4": np.array([0.045, 0.067, 0.109]),
        "camera5": np.array([0.065, 0.059, 0.110]),
        "can1": np.array([0.061, 0.115, 0.061]),
        "can2": np.array([0.084, 0.079, 0.084]),
        "can3": np.array([0.063, 0.113, 0.063]),
        "can4": np.array([0.063, 0.113, 0.063]),
        "bottle1": np.array([0.057, 0.115, 0.057]),
        "bottle2": np.array([0.061, 0.122, 0.039]),
        "bottle3": np.array([0.073, 0.098, 0.073]),
        "bottle4": np.array([0.062, 0.096, 0.062]),
        "bowl1": np.array([0.095, 0.048, 0.095]),
        "bowl2": np.array([0.093, 0.048, 0.093]),
        "mug2": np.array([0.125, 0.095, 0.086]),
        "mug3": np.array([0.092, 0.084, 0.070]),
        "mug4": np.array([0.098, 0.080, 0.068]),
    }
    return shape_xyzs


def sym_axes():
    sym_axes = {
        "camera1": None,
        "camera2": None,
        "camera3": None,
        "camera4": None,
        "camera5": None,
        "can1": "y",
        "can2": "y",
        "can3": "y",
        "can4": "y",
        "bottle1": "y",
        "bottle2": "y",
        "bottle3": "y",
        "bottle4": "y",
        "bowl1": "y",
        "bowl2": "y",
        "mug2": None,
        "mug3": None,
        "mug4": None,
    }
    return sym_axes
