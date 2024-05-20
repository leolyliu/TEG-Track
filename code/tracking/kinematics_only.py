import os
from os.path import join, dirname, abspath
import sys
BASEPATH = dirname(abspath(__file__))
sys.path.insert(0, join(BASEPATH, '..'))
import argparse
import numpy as np
import cv2
import pickle
import torch
from transforms3d.euler import euler2mat, mat2euler, euler2quat, quat2mat
from utils.get_data import get_real_initial_state, get_contact_info, read_from_pkl
from utils.pose import pose_dict2mat, rotation_lerp
from utils.kinematic import pred_rotation_from_omega, pred_velocity_from_pose
from utils.slip_detection import fit_affine_transformation
from utils.visualization import bbox_visualization
from optimizer.velocity_estimator import solve_object_velocities
from tactile_learning.model import TactilePoseNet
from tactile_learning.utils import train_test_split, shape_xyzs, sym_axes
from statistics.statistics_posediff import eval


def process_data(last_data, data, device, source="track_by_learning"):  # align with the dataset
    model_input = {}
    if source == "track_by_learning":
        for sensor in ["left", "right"]:
            img0 = last_data[sensor]["rgb"]
            img1 = data[sensor]["rgb"]
            img0 = cv2.resize(img0, (384, 288))
            img1 = cv2.resize(img1, (384, 288))
            delta_img = (img1.astype(np.float32) - img0.astype(np.float32)) / 255
            model_input[sensor] = torch.as_tensor(delta_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    elif source == "track_by_mix_strategy":
        for sensor in ["left", "right"]:
            img0 = last_data[sensor + "_rgb"].copy()
            img1 = data[sensor + "_rgb"].copy()
            img0 = cv2.resize(img0, (384, 288))
            img1 = cv2.resize(img1, (384, 288))
            delta_img = (img1.astype(np.float32) - img0.astype(np.float32)) / 255
            model_input[sensor] = torch.as_tensor(delta_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    else:
        raise NotImplementedError
    
    return model_input


def has_slippage(last_data, data, threshold=10.0):
    if last_data is None:
        return False
    
    left_depth = data["left_depth"].astype(np.float32) / 100000
    left_pts_int = data["left_pts"].astype(np.int32)
    flag = (0.0002 < left_depth[left_pts_int[1], left_pts_int[0]]) & (left_depth[left_pts_int[1], left_pts_int[0]] < 0.049)
    left_contact_pts = data["left_pts"][:, flag]
    left_slip_number = 0
    if (left_contact_pts.shape[1] >= 3) and (last_data["left_pts"].shape[1] >= 3):
        left_slip_number = fit_affine_transformation(left_contact_pts, last_data["left_pts"])

    right_depth = data["right_depth"].astype(np.float32) / 100000
    right_pts_int = data["right_pts"].astype(np.int32)
    flag = (0.0002 < right_depth[right_pts_int[1], right_pts_int[0]]) & (right_depth[right_pts_int[1], right_pts_int[0]] < 0.049)
    right_contact_pts = data["right_pts"][:, flag]
    right_slip_number = 0
    if (right_contact_pts.shape[1] >= 3) and (last_data["right_pts"].shape[1] >= 3):
        right_slip_number = fit_affine_transformation(right_contact_pts, last_data["right_pts"])

    return max(left_slip_number, right_slip_number) > threshold


def track_by_mix_strategy(sequence_path, ids, model, shape_xyz, sym_axis, save_path, save_kinematics_path=None, span=10):
    os.makedirs(save_path, exist_ok=True)
    sensors = ["left", "right"]
    last_data = None
    pred_ids = []
    pred_kinematics = []
    pred_poses = []
    gt_poses = []

    status = {
        "status": "optimization",
        "start_frame": None,
        "start_data": None,
    }

    for idx in ids:
        pred_ids.append(idx)
        _, pose = get_real_initial_state(sequence_path, idx=idx)
        data = {
            "left_rgb": cv2.imread(join(sequence_path, "left_rgb", str(idx).zfill(4) + ".png"), cv2.IMREAD_UNCHANGED),
            "right_rgb": cv2.imread(join(sequence_path, "right_rgb", str(idx).zfill(4) + ".png"), cv2.IMREAD_UNCHANGED),
            "left_model_matrix": np.loadtxt(join(sequence_path, "left_pad_pose", str(idx).zfill(4) + ".txt")),  # camera pose
            "right_model_matrix": np.loadtxt(join(sequence_path, "right_pad_pose", str(idx).zfill(4) + ".txt")),  # camera pose
            "left_depth": cv2.imread(join(sequence_path, "left_depth", str(idx).zfill(4) + ".png"), cv2.CV_16UC1),
            "right_depth": cv2.imread(join(sequence_path, "right_depth", str(idx).zfill(4) + ".png"), cv2.CV_16UC1),
            "left_pts": np.loadtxt(join(sequence_path, "left_marker_point_positions", str(idx).zfill(4) + ".txt")),
            "right_pts": np.loadtxt(join(sequence_path, "right_marker_point_positions", str(idx).zfill(4) + ".txt")),
            "gt_pose": pose,
        }
        gt_poses.append(pose)

        if idx == ids[0]:
            last_data = data.copy()
            pred_poses.append(pose)
            pred_kinematics.append(None)
            continue

        # add physics data
        physics = read_from_pkl(join(sequence_path, "precomputed_object_contact_information", str(idx).zfill(4) + ".pkl"))
        data["left_contact_points"] = physics["left_contact_points"]
        data["right_contact_points"] = physics["right_contact_points"]
        data["left_mean_contact_velocities"] = physics["left_mean_contact_velocities"]
        data["right_mean_contact_velocities"] = physics["right_mean_contact_velocities"]
        data["dt"] = physics["dt"]

        if status["status"] == "optimization":
            if has_slippage(last_data, data):
                # switch mode
                status["status"] = "mix"
                status["start_frame"] = idx - 1
                status["start_data"] = last_data.copy()

        # use optimization to predict pose differences
        dt = data["dt"]
        lcp, lcv, rcp, rcv = get_contact_info(data, add_noise=False)
        if (lcp is None) or (len(lcp.shape) == 1) or (lcp.shape[0] == 0):
            lcp = lcp.reshape(0, 3)
            lcv = lcv.reshape(0, 3)
        if (rcp is None) or (len(rcp.shape) == 1) or (rcp.shape[0] == 0):
            rcp = rcp.reshape(0, 3)
            rcv = rcv.reshape(0, 3)
        init_t = pred_poses[-1]["translation"].reshape(3)
        contact_points = np.concatenate((lcp, rcp))
        contact_velocities = np.concatenate((lcv, rcv))
        v, omega, curr_t = solve_object_velocities(contact_points, contact_velocities, init_t, iter=10)
        if np.linalg.norm(omega) < 1e-8:
            omega = np.array([0, 0, 1e-7])
        pred_kinematics.append({
            "dt": dt,
            "v": v.reshape(3),
            "omega": omega.reshape(3),
            "t": pred_poses[-1]["translation"].reshape(3),
        })
        t0 = pred_poses[-1]["translation"].reshape(3)
        r0 = pred_poses[-1]["rotation"]
        t = t0 + v * dt
        R = pred_rotation_from_omega(r0, omega, dt)
        pred_poses.append({"translation": t.reshape(3, 1), "rotation": R})

        if status["status"] == "mix":
            if idx - status["start_frame"] == span:
                # individually predict current pose by each tactile sensor
                model_input = process_data(status["start_data"], data, device="cuda:0", source="track_by_mix_strategy")
                last_pose = pose_dict2mat(pred_poses[-span-1])
                pose_proposals = []
                for sensor in sensors:
                    pred = model(model_input[sensor]).detach().cpu().numpy()[0]
                    T = np.eye(4)
                    T[:3, :3] = euler2mat(pred[0] / 5, pred[1] / 5, pred[2] / 5)
                    T[:3, 3] = pred[3:6] / 100
                    pose = data[sensor + "_model_matrix"] @ T @ np.linalg.inv(data[sensor + "_model_matrix"]) @ last_pose
                    pose_proposals.append(pose)

                # interpolation between result_left and result_right
                assert len(pose_proposals) == 2
                t = (pose_proposals[0][:3, 3:] + pose_proposals[1][:3, 3:]) / 2
                ai, aj, ak = mat2euler(pose_proposals[0][:3, :3])
                q0 = euler2quat(ai, aj, ak)
                bi, bj, bk = mat2euler(pose_proposals[1][:3, :3])
                q1 = euler2quat(bi, bj, bk)
                if np.dot(q0, q1) < np.dot(q0, -q1):
                    q1 = -q1
                q = (q0 + q1) / 2
                q /= np.linalg.norm(q)
                R = quat2mat(q)

                # rotation+translation interpolation
                for i in range(span):
                    pred_poses[-span+i]["rotation"] = rotation_lerp(pred_poses[-span-1]["rotation"], R, (i + 1) / span)
                    pred_poses[-span+i]["translation"] = (1 - (i + 1) / span) * pred_poses[-span-1]["translation"] + (i + 1) / span * t
                    v, _, omega = pred_velocity_from_pose(pred_poses[-span+i]["translation"].reshape(3, 1), pred_poses[-span+i-1]["translation"].reshape(3, 1), pred_poses[-span+i]["rotation"], pred_poses[-span+i-1]["rotation"], pred_kinematics[-span+i]["dt"])
                    pred_kinematics[-span+i]["v"] = v.reshape(3)
                    pred_kinematics[-span+i]["omega"] = omega.reshape(3)
                    pred_kinematics[-span+i]["t"] = ((1 - i / span) * pred_poses[-span-1]["translation"] + i / span * t).reshape(3)
                
                # switch mode
                status["status"] = "optimization"

        last_data = data.copy()
    
    # save results
    translations, rotations = [], []
    for p in pred_poses:
        translations.append(p["translation"].reshape(3))
        rotations.append(p["rotation"])
    results = {
        "ids": pred_ids,
        "translations": np.float32(translations),
        "rotations": np.float32(rotations),
    }
    pickle.dump(results, open(join(save_path, "tracking_results_final_opt.pkl"), "wb"))
    if not save_kinematics_path is None:
        os.makedirs(save_kinematics_path, exist_ok=True)
        pickle.dump(pred_kinematics, open(join(save_kinematics_path, "kinematics.pkl"), "wb"))

    # statistics
    eval(pred_poses, gt_poses, np.arange(ids[0], ids[-1], 1), save_path, sym_axis=sym_axis, output_path=join(save_path, "eval.txt"))

    # visualization
    bbox_visualization("", save_path, pred_poses, gt_poses=gt_poses, path=sequence_path, frame_size=(1280, 720), scaling=False, start_idx=ids[0], view_mode="3rd_view", fps=30, shape_xyz=shape_xyz)


# serve for process_real
def inference_kinematics_taclearn(sequence_path, start_idx, rawdatas, span=10, model_path=join(BASEPATH, "../tactile_learning/checkpoints/epoch_30.pth")):
    sensors = ["left", "right"]
    last_data = None
    pred_poses = []

    model = TactilePoseNet()
    model.to("cuda:0")
    model.load_state_dict(torch.load(model_path), strict=True)

    status = {
        "status": "optimization",
        "start_frame": None,
        "start_data": None,
    }

    N = len(rawdatas)
    for idx in range(N):
        rawdata = rawdatas[idx]
        data = {
            "left_rgb": cv2.imread(join(sequence_path, "left_rgb", str(idx+start_idx).zfill(4) + ".png"), cv2.IMREAD_UNCHANGED),
            "right_rgb": cv2.imread(join(sequence_path, "right_rgb", str(idx+start_idx).zfill(4) + ".png"), cv2.IMREAD_UNCHANGED),
            "left_model_matrix": np.loadtxt(join(sequence_path, "left_pad_pose", str(idx+start_idx).zfill(4) + ".txt")),  # camera pose
            "right_model_matrix": np.loadtxt(join(sequence_path, "right_pad_pose", str(idx+start_idx).zfill(4) + ".txt")),  # camera pose
            "left_depth": cv2.imread(join(sequence_path, "left_depth", str(idx+start_idx).zfill(4) + ".png"), cv2.CV_16UC1),
            "right_depth": cv2.imread(join(sequence_path, "right_depth", str(idx+start_idx).zfill(4) + ".png"), cv2.CV_16UC1),
            "left_pts": np.loadtxt(join(sequence_path, "left_marker_point_positions", str(idx+start_idx).zfill(4) + ".txt")),
            "right_pts": np.loadtxt(join(sequence_path, "right_marker_point_positions", str(idx+start_idx).zfill(4) + ".txt")),
        }

        if idx == 0:
            last_data = data.copy()
            _, pose = get_real_initial_state(sequence_path, idx=idx+start_idx)
            pred_poses.append(pose)
            continue

        # add physics data
        data["left_contact_points"] = np.array(rawdata["left_contact_points"])
        data["right_contact_points"] = np.array(rawdata["right_contact_points"])
        data["left_mean_contact_velocities"] = np.array(rawdata["left_mean_contact_velocities"])
        data["right_mean_contact_velocities"] = np.array(rawdata["right_mean_contact_velocities"])
        data["dt"] = rawdata["dt"]

        if status["status"] == "optimization":
            if has_slippage(last_data, data):
                # switch mode
                status["status"] = "mix"
                status["start_frame"] = idx - 1
                status["start_data"] = last_data.copy()

        # use optimization to predict delta_pose (original "physics_only")
        dt = data["dt"]
        lcp, lcv, rcp, rcv = get_contact_info(data, add_noise=False)
        if (lcp is None) or (len(lcp.shape) == 1) or (lcp.shape[0] == 0):
            lcp = lcp.reshape(0, 3)
            lcv = lcv.reshape(0, 3)
        if (rcp is None) or (len(rcp.shape) == 1) or (rcp.shape[0] == 0):
            rcp = rcp.reshape(0, 3)
            rcv = rcv.reshape(0, 3)
        init_t = pred_poses[-1]["translation"].reshape(3)
        contact_points = np.concatenate((lcp, rcp))
        contact_velocities = np.concatenate((lcv, rcv))
        v, omega, curr_t = solve_object_velocities(contact_points, contact_velocities, init_t, iter=10)
        if np.linalg.norm(omega) < 1e-8:
            omega = np.array([0, 0, 1e-7])
        t0 = pred_poses[-1]["translation"].reshape(3)
        r0 = pred_poses[-1]["rotation"]
        t = t0 + v * dt
        R = pred_rotation_from_omega(r0, omega, dt)
        pred_poses.append({"translation": t.reshape(3, 1), "rotation": R})

        if status["status"] == "mix":
            if idx - status["start_frame"] == span:
                # individually predict current pose by each tactile sensor
                model_input = process_data(status["start_data"], data, device="cuda:0", source="track_by_mix_strategy")
                last_pose = pose_dict2mat(pred_poses[-span-1])
                pose_proposals = []
                for sensor in sensors:
                    pred = model(model_input[sensor]).detach().cpu().numpy()[0]
                    T = np.eye(4)
                    T[:3, :3] = euler2mat(pred[0] / 5, pred[1] / 5, pred[2] / 5)
                    T[:3, 3] = pred[3:6] / 100
                    pose = data[sensor + "_model_matrix"] @ T @ np.linalg.inv(data[sensor + "_model_matrix"]) @ last_pose
                    pose_proposals.append(pose)

                # interpolation between result_left and result_right
                assert len(pose_proposals) == 2
                t = (pose_proposals[0][:3, 3:] + pose_proposals[1][:3, 3:]) / 2
                ai, aj, ak = mat2euler(pose_proposals[0][:3, :3])
                q0 = euler2quat(ai, aj, ak)
                bi, bj, bk = mat2euler(pose_proposals[1][:3, :3])
                q1 = euler2quat(bi, bj, bk)
                if np.dot(q0, q1) < np.dot(q0, -q1):
                    q1 = -q1
                q = (q0 + q1) / 2
                q /= np.linalg.norm(q)
                R = quat2mat(q)
                
                # rotation+translation interpolation
                for i in range(span):
                    pred_poses[-span+i]["rotation"] = rotation_lerp(pred_poses[-span-1]["rotation"], R, (i + 1) / span)
                    # pred_poses[-span+i]["translation"] = (1 - (i + 1) / span) * pred_poses[-span-1]["translation"] + (i + 1) / span * t
                
                # switch mode
                status["status"] = "optimization"

        last_data = data.copy()
    
    # save results
    translations, rotations = [], []
    for p in pred_poses:
        translations.append(p["translation"].reshape(3))
        rotations.append(p["rotation"])

    kinematics = [{"v": None, "omega": None, "t": None}]
    for idx in range(N-1):
        v, _, omega = pred_velocity_from_pose(translations[idx+1].reshape(3, 1), translations[idx].reshape(3, 1), rotations[idx+1], rotations[idx], rawdatas[idx+1]["dt"])
        kinematics.append({"v": v.reshape(3), "omega": omega.reshape(3), "t": translations[idx].reshape(3)})
    return kinematics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="/nas/datasets/Visual_Tactile_Tracking_Dataset_Released")
    parser.add_argument('--tracking_save_dir', type=str, default="./tracking_results_kinematics-only")
    parser.add_argument('--kinematics_save_dir', type=str, default="./object_kinematic_states")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    tracking_save_dir = args.tracking_save_dir
    kinematics_save_dir = args.kinematics_save_dir

    _, video_names = train_test_split()
    shape_xyz_dict = shape_xyzs()
    sym_axis_dict = sym_axes()

    model = TactilePoseNet()
    model.to("cuda:0")
    model.load_state_dict(torch.load(join(BASEPATH, "../tactile_learning/checkpoints/epoch_30.pth")), strict=True)

    for video_name in video_names:
        obj_name = video_name.split("_")[1]
        obj_category = obj_name[:-1]
        category_dir = join(dataset_dir, obj_category)
        sequence_list = []
        for sequence_name in os.listdir(category_dir):
            if sequence_name.startswith(video_name):
                sequence_list.append(sequence_name)
        sequence_list.sort()
        
        for sequence_name in sequence_list:
            ids = list(np.arange(0, 401))
            save_path = join(tracking_save_dir, sequence_name)
            save_kinematics_path = join(kinematics_save_dir, sequence_name)
            track_by_mix_strategy(sequence_path=join(category_dir, sequence_name), ids=ids, model=model, shape_xyz=shape_xyz_dict[obj_name], sym_axis=sym_axis_dict[obj_name], save_path=save_path, save_kinematics_path=save_kinematics_path, span=10)
