import os
import pickle
import open3d as o3d
import cv2
import numpy as np
from transforms3d.euler import mat2euler, euler2mat
from utils.TSDF import pcd_denoise_and_downsample


def filter_depth_from_mask(file_depth, file_mask):
    depth_raw = cv2.imread(file_depth, cv2.IMREAD_UNCHANGED)
    mask_raw = cv2.imread(file_mask, cv2.IMREAD_UNCHANGED)
    mask = mask_raw[:, :, 0]
    depth_raw[mask > 127] = 0
    new_file_depth = file_depth.replace("depth", "depthwithmask")
    cv2.imwrite(new_file_depth, depth_raw)
    depth_raw = o3d.io.read_image(new_file_depth)
    return depth_raw


def filter_depth_from_mask_input_data(depth_raw, mask_raw, new_file_depth):
    mask = mask_raw[:, :, 0]
    depth_raw[mask > 127] = 0
    cv2.imwrite(new_file_depth, depth_raw)
    depth_raw = o3d.io.read_image(new_file_depth)
    return depth_raw


def read_intrinsics_from_txt(file_path):
    x = []
    with open(file_path, "r") as f:
        for line in f:
            values = line.strip().split(" ")
            if len(values) != 3:
                continue
            x.append([float(values[0]), float(values[1]), float(values[2])])
    x = np.array(x)
    assert x.shape == (3, 3)
    return x


def read_extrinsics_from_txt(file_path):
    x = []
    with open(file_path, "r") as f:
        for line in f:
            values = line.strip().split(" ")
            if len(values) != 4:
                continue
            x.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
    x = np.array(x)
    assert x.shape == (4, 4)
    return x


def read_pose_from_txt(file_path):
    x = []
    with open(file_path, "r") as f:
        for line in f:
            values = line.strip().split(" ")
            if len(values) != 4:
                continue
            x.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
    x = np.array(x)
    assert x.shape == (4, 4)

    pose = {"translation": x[:3, 3], "rotation": x[:3, :3]}
    return pose


def read_model_matrix_from_txt(file_path):
    x = []
    with open(file_path, "r") as f:
        for line in f:
            values = line.strip().split(" ")
            if len(values) != 4:
                continue
            x.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
    x = np.array(x)
    assert x.shape == (4, 4)
    return x


def read_from_pkl(file_pose):
    pose_raw = pickle.load(open(file_pose, 'rb'))
    return pose_raw


def read_from_npy(file_pose):
    pose_raw = np.load(file_pose)
    pose = {
        "rotation": pose_raw[:3, :3],
        "translation": pose_raw[:3, 3].reshape(3, 1),
    }
    return pose


def check_depth2pts(depth, intrinsic, extrinsic):
    depth = depth / 10000
    v, u = np.where(depth > 0)
    uv = np.vstack((u + 0.5, v + 0.5, np.ones(u.shape[0])))
    uv = np.matmul(np.linalg.inv(intrinsic), uv)
    cp = uv * np.reshape(depth[depth > 0], (1, -1))
    r = extrinsic[:3, :3]
    t = extrinsic[:3, 3:4]
    r_inv = np.linalg.inv(r)
    wp = np.matmul(r_inv, cp - t).transpose()
    return wp


def get_tactile_points(depth, model_matrix, use_noise=False, Pixmm=5.9259259259e-5):
    depth = depth / 100000
    H, W = depth.shape
    flag = (0 < depth) & (depth < 0.048)
    pY, pX = np.where(flag)
    X = (pX - W / 2) * Pixmm
    Y = (H / 2 - pY) * Pixmm
    if not use_noise:
        Z = - depth[flag]
    else:
        Z = depth[flag] - 0.046
    camera_points = np.vstack((X, Y, Z)).T
    world_points = np.matmul(camera_points, model_matrix[:3, :3].T) + model_matrix[:3, 3]
    return world_points


def get_all_data_from_files(root_dir, idx, start_idx=0, view_mode=None, real=False):
    
    assert view_mode == "3rd_view"
    
    file_visual_extrinsic = os.path.join(root_dir, "camera_params", "visual_camera_extrinsic.txt")
    file_visual_intrinsic = os.path.join(root_dir, "camera_params", "visual_camera_intrinsic.txt")
    file_left_model_matrix = os.path.join(root_dir, "left_pad_pose", str(idx).zfill(4) + ".txt")
    file_right_model_matrix = os.path.join(root_dir, "right_pad_pose", str(idx).zfill(4) + ".txt")

    file_pcd = os.path.join(root_dir, "visual_point_cloud", str(idx).zfill(4) + ".ply")
    file_left_pcd = os.path.join(root_dir, "left_point_cloud", str(idx).zfill(4) + ".ply")
    file_right_pcd = os.path.join(root_dir, "right_point_cloud", str(idx).zfill(4) + ".ply")

    # get visual input
    visual_intrinsic = read_intrinsics_from_txt(file_visual_intrinsic)
    visual_extrinsic = read_extrinsics_from_txt(file_visual_extrinsic)
    visual_points = np.array(o3d.io.read_point_cloud(file_pcd).points)

    # get tactile input
    left_model_matrix = read_model_matrix_from_txt(file_left_model_matrix)
    left_points = np.array(o3d.io.read_point_cloud(file_left_pcd).points)
    right_model_matrix = read_model_matrix_from_txt(file_right_model_matrix)
    right_points = np.array(o3d.io.read_point_cloud(file_right_pcd).points)
    tactile_points = np.concatenate((left_points, right_points), axis=0)

    all_points = np.concatenate((visual_points, tactile_points), axis=0)

    if int(idx) == int(start_idx):
        return visual_points, tactile_points, all_points, visual_intrinsic, visual_extrinsic, left_model_matrix, right_model_matrix
    else:
        file_physics = os.path.join(root_dir, "precomputed_object_contact_information", str(idx).zfill(4) + ".pkl")
        physics = read_from_pkl(file_physics)
        return physics, visual_points, tactile_points, all_points, visual_intrinsic, visual_extrinsic, left_model_matrix, right_model_matrix


def get_contact_info(rawdata, add_noise=True, noise_p=0.003, noise_v=1.4):
    lcp = np.array(rawdata["left_contact_points"])
    lcv = np.array(rawdata["left_mean_contact_velocities"])
    rcp = np.array(rawdata["right_contact_points"])
    rcv = np.array(rawdata["right_mean_contact_velocities"])
    if add_noise:
        lcp += np.random.normal(0, noise_p, lcp.shape)
        lcv *= np.random.uniform(1 / noise_v, noise_v, lcv.shape)
        rcp += np.random.normal(0, noise_p, rcp.shape)
        rcv *= np.random.uniform(1 / noise_v, noise_v, rcv.shape)
    return lcp, lcv, rcp, rcv


def get_real_data(root_dir, frame_idx, start_idx=0, view_mode=None, need_pcd=True):
    if int(frame_idx) == int(start_idx):
        visual_points, tactile_points, all_points, visual_intrinsic, visual_extrinsic, left_model_matrix, right_model_matrix = get_all_data_from_files(root_dir, frame_idx, start_idx=start_idx, view_mode=view_mode)
        data = {
            "extrinsic": visual_extrinsic,
            "visual_points": visual_points,
            "tactile_points": tactile_points,
            "all_points": all_points,
        }
        return data

    physics, visual_points, tactile_points, all_points, visual_intrinsic, visual_extrinsic, left_model_matrix, right_model_matrix = get_all_data_from_files(root_dir, frame_idx, start_idx=start_idx, view_mode=view_mode)
    if need_pcd:
        visual_points, tactile_points, all_points = pcd_denoise_and_downsample(visual_points, tactile_points)
    dt = physics["dt"]
    left_contact_points = physics["left_contact_points"]
    right_contact_points = physics["right_contact_points"]
    left_mean_contact_velocities = physics["left_mean_contact_velocities"]
    right_mean_contact_velocities = physics["right_mean_contact_velocities"]

    data = {
        "dt": dt,
        "extrinsic": visual_extrinsic,
        "left_contact_points": left_contact_points,
        "right_contact_points": right_contact_points,
        "left_mean_contact_velocities": left_mean_contact_velocities,
        "right_mean_contact_velocities": right_mean_contact_velocities,
        "visual_points": visual_points,
        "tactile_points": tactile_points,
        "all_points": all_points,
    }
    return data


def get_real_initial_state(root_dir, idx=0, noisy_initial=False, noise_scale=1.0):
    pose = read_from_npy(os.path.join(root_dir, "object_pose", str(idx).zfill(4) + ".npy"))

    t_gt = pose["translation"]
    ai, aj, ak = mat2euler(pose["rotation"])
    euler_gt = np.array((ai, aj, ak))
    N = 6
    state = np.zeros((N, 1))
    state[0:3] = t_gt.reshape(3, 1)
    state[3:6] = euler_gt.reshape(3, 1)
    if noisy_initial:
        state[0:3] += noise_scale * np.random.normal(0, 0.001, (3, 1))
        state[3:6] += noise_scale * np.random.normal(0, 0.1, (3, 1))
    state = state.reshape(N)
    return state, pose
