import os
import sys
from os.path import join, dirname, abspath, isfile
import time
BASEPATH = dirname(abspath(__file__))
sys.path.insert(0, join(BASEPATH, '..'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from optimizer.torch_geometry_optimizer import TorchOptimizerGeometry
from statistics.statistics_posediff import eval
from utils.get_data import get_real_data, get_real_initial_state, get_contact_info
from utils.TSDF import load_model
from utils.kinematic import pred_rotation_from_omega
from utils.pose import normalized_euler, pose_world2cam, pose_cam2world
from utils.visualization import bbox_visualization
from utils.save_data import save_tracking_results
from utils.dataset_info import get_dataset_info
from transforms3d.euler import mat2euler, euler2mat
from process.stepwise import optimize_multigeo
from optimizer.velocity_estimator import solve_object_velocities, cross
from tactile_learning.utils import train_test_split
from tracking.kinematics_only import inference_kinematics_taclearn
import open3d as o3d


def draw_scalar(pred, gt, name=""):
    x = np.arange(0, len(pred))
    plt.clf()
    plt.plot(x, pred[:, 0], color="red", label="pred 0")
    plt.plot(x, gt[:, 0], color="blue", label="gt 0")
    plt.plot(x, pred[:, 1], color="green", label="pred 1")
    plt.plot(x, gt[:, 1], color="yellow", label="gt 1")
    plt.plot(x, pred[:, 2], color="black", label="pred 2")
    plt.plot(x, gt[:, 2], color="cyan", label="gt 2")
    plt.legend()
    plt.savefig(name)


def compute_loss(v, omega, t, cps, cvs):
    v = v.reshape(3, 1)
    omega = omega.reshape(3, 1)
    t = t.reshape(3, 1)
    loss = 0
    cnt = 0
    assert cps.shape == cvs.shape
    for i in range(cps.shape[0]):
        p = cps[i].reshape(3, 1)
        pv = cvs[i].reshape(3, 1)
        pv_pred = v + cross(omega, (p - t))
        cnt += 1
        loss += np.linalg.norm(pv - pv_pred, ord=1)
    loss /= cnt
    return loss


def optimize_frame_geometry_real(model_points, rawdatas, last_state, device="cpu"):
    N = 6  # optimize position, rotation

    scale = 1.0
    visual_points = rawdatas[1]["visual_points"]
    tactile_points = rawdatas[1]["tactile_points"]
    extrinsic0 = rawdatas[0]["extrinsic"]
    extrinsic1 = rawdatas[1]["extrinsic"]

    # init state
    t0 = last_state[0:3].reshape(3, 1)
    r0 = euler2mat(last_state[3], last_state[4], last_state[5])

    r0, t0 = pose_world2cam(r0, t0, extrinsic0)
    r0, t0 = pose_cam2world(r0, t0, extrinsic1)
    init_state = np.zeros(N)
    init_state[0:3] = t0.reshape(3)
    ai, aj, ak = mat2euler(r0)
    init_state[3:6] = np.array((ai, aj, ak))

    optimizer = TorchOptimizerGeometry(visual_points=visual_points, tactile_points=tactile_points, extrinsic=extrinsic1, model_points=model_points, scale=scale, init_state=init_state, device=device)
    optimizer.solve(epoch=20)
    answer_state, answer_energy = optimizer.get_answer()
    
    print("energy:", answer_energy)
    return answer_state


def stepwise_taclearn(root_dir, ids, view_mode, result_path, model_path, device, N_frames=5, use_completed_point_cloud=True, completed_point_cloud=None, sym_axis=None, ground_truth=False, use_gt_velocities=False, shape_xyz=None, gt_scale=1.0, kinematics_save_dir=""):
    os.makedirs(result_path, exist_ok=True)
    rawdatas = []
    pred_states = []
    pred_poses = []
    model_points = None

    for idx in ids:
        # get data
        data = get_real_data(root_dir, str(idx), start_idx=ids[0], view_mode=view_mode)
        rawdatas.append(data)
    
    assert use_gt_velocities == False

    if not isfile(join(kinematics_save_dir, "kinematics.pkl")):
        print("############ start pre-processing object kinematic states ... ############")
        kinematics = inference_kinematics_taclearn(root_dir, ids[0], rawdatas)
        print("############ finish pre-processing object kinematic states !!! ############")
    else:
        print("############ start loading object kinematic states ... ############")
        kinematics = pickle.load(open(join(kinematics_save_dir, "kinematics.pkl"), "rb"))
        print("############ finish loading object kinematic states !!! ############")

    print("############ start estimating object poses ... ############")
    idx = ids[0]
    while idx < ids[0] + len(rawdatas):
        if idx == ids[0]:
            # get geometry, initial pose
            state, pose = get_real_initial_state(root_dir, idx=idx)
            pred_states.append(state)
            pred_poses.append(pose)
            if use_completed_point_cloud:
                model_points = completed_point_cloud
            else:
                model_points = load_model(model_path, K_points=2048, scale=gt_scale)
            idx += 1
            continue
        if len(rawdatas)+ids[0]-idx >= N_frames:
            print("frame {} use multigeo!!!".format(idx))
            # optimize velocity only
            delta_time = []
            for i in range(N_frames):
                dt = rawdatas[idx - ids[0] + i]['dt']
                delta_time.append(dt)
                lcp, lcv, rcp, rcv = get_contact_info(rawdatas[idx - ids[0] + i], add_noise=False)
                if (lcp is None) or (len(lcp.shape) == 1) or (lcp.shape[0] == 0):
                    lcp = lcp.reshape(0, 3)
                    lcv = lcv.reshape(0, 3)
                if (rcp is None) or (len(rcp.shape) == 1) or (rcp.shape[0] == 0):
                    rcp = rcp.reshape(0, 3)
                    rcv = rcv.reshape(0, 3)

                init_t = pred_states[idx - ids[0] + i - 1][0:3]
                contact_points = np.concatenate((lcp, rcp))
                contact_velocities = np.concatenate((lcv, rcv))

                v, omega, v_t = kinematics[idx - ids[0] + i]["v"], kinematics[idx - ids[0] + i]["omega"], kinematics[idx - ids[0] + i]["t"].reshape(3)  # v_t: 定义线速度使用的刚体坐标系原点
                if np.linalg.norm(omega) < 1e-8:
                    omega = np.array([0, 0, 1e-7])

                t0 = pred_states[idx - ids[0] + i - 1][0:3]
                r0 = euler2mat(pred_states[idx - ids[0] + i - 1][3], pred_states[idx - ids[0] + i - 1][4], pred_states[idx - ids[0] + i - 1][5])
                t = t0 + (v + cross(omega, t0 - v_t)) * dt
                r = pred_rotation_from_omega(r0, omega, dt)
                ex, ey, ez = mat2euler(r)
                euler = np.array([ex, ey, ez])
                pred_states.append(np.concatenate((t, euler, v, omega)))
                pose = {"translation": t.reshape(3, 1), "rotation": r}
                pred_poses.append(pose)

            # constrained multi-geometry optimization
            start_time = time.time()
            states = optimize_multigeo(N_frames, model_points, delta_time, 1.0, rawdatas[idx-ids[0]:idx-ids[0]+N_frames], pred_states[idx-ids[0]:idx-ids[0]+N_frames], None, device)
            print("geometric-kinematic optimization FPS =", N_frames / (time.time() - start_time))

            # update poses
            pred_states[idx-ids[0]:idx-ids[0]+N_frames] = states
            for i in range(idx-ids[0], idx-ids[0]+N_frames):
                state = pred_states[i]
                pred_poses[i] = {"translation": state[:3].reshape(3, 1), "rotation": euler2mat(state[3], state[4], state[5])}
            idx += N_frames

        else:
            # geometry-only
            print("frame {} use geometry!!!".format(idx))
            last_state = pred_states[idx - ids[0] - 1][0:6]
            state = optimize_frame_geometry_real(model_points, rawdatas[idx-ids[0]-1:idx-ids[0]+1], last_state, device)
            state[3], state[4], state[5] = normalized_euler(state[3], state[4], state[5])
            lcp, lcv, rcp, rcv = get_contact_info(rawdatas[idx - ids[0]], add_noise=False)
            init_t = pred_states[idx - ids[0] - 1][0:3]
            contact_points = np.concatenate((lcp, rcp))
            contact_velocities = np.concatenate((lcv, rcv))
            v, omega, _ = solve_object_velocities(contact_points, contact_velocities, init_t, iter=10)
            pred_states.append(np.concatenate((state, v, omega)))
            pose = {"translation": state[:3].reshape(3, 1), "rotation": euler2mat(state[3], state[4], state[5])}
            pred_poses.append(pose)
            idx += 1
    print("############ finish estimating object poses !!! ############")

    # save tracking results
    save_path = os.path.join(result_path, "tracking_results_final_opt.pkl")
    save_tracking_results(pred_states, save_path, save_dynamics=False, save_constants=False)

    # quantitative evaluation
    gt_poses = None
    if ground_truth:
        gt_poses = []
        last_valid_pose = None
        for idx in ids:
            state, gt_pose = get_real_initial_state(root_dir, idx=idx)
            if gt_pose["translation"][0] > 100:  # the motion capture system (used to construct the dataset) loses track at this frame
                assert not last_valid_pose is None
                gt_pose = last_valid_pose
            else:
                last_valid_pose = gt_pose
            gt_poses.append(gt_pose)
        eval(pred_poses, gt_poses, ids, result_path, sym_axis, output_path=os.path.join(result_path, "eval.txt"))

    # bbox visualization
    bbox_visualization(model_path, result_path, pred_poses, gt_poses=gt_poses, path=root_dir, frame_size=(1280, 720), scaling=False, start_idx=ids[0], view_mode=view_mode, fps=30, shape_xyz=shape_xyz)


def solve(sequence_name, sequence_dir, save_dir, kinematics_save_root, ids, N_frames, use_completed_point_cloud=True, completed_point_cloud=None, sym_axis=None, shape_xyz=None, gt_scale=0.14, model_path=""):
    view_mode = '3rd_view'
    device = "cpu"

    output_folder_step = join(save_dir, sequence_name)
    stepwise_taclearn(sequence_dir, ids, view_mode, result_path=output_folder_step, model_path=model_path, device=device, N_frames=N_frames, use_completed_point_cloud=use_completed_point_cloud, completed_point_cloud=completed_point_cloud, sym_axis=sym_axis, ground_truth=True, use_gt_velocities=False, shape_xyz=shape_xyz, gt_scale=gt_scale, kinematics_save_dir=join(kinematics_save_root, sequence_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="/nas/datasets/Visual_Tactile_Tracking_Dataset_Released")
    parser.add_argument('--completed_point_cloud_dir', type=str, default="/nas/datasets/Visual_Tactile_Tracking_Dataset_Released/preprocessed_completed_point_cloud_data")
    parser.add_argument('--tracking_save_dir', type=str, default="./tracking_results_TEG-Track(ShapeAlign)")
    parser.add_argument('--kinematics_save_dir', type=str, default="./object_kinematic_states")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    completed_point_cloud_dir = args.completed_point_cloud_dir
    tracking_save_dir = args.tracking_save_dir
    kinematics_save_dir = args.kinematics_save_dir
    
    video_names, sym_axis_list, shape_xyz_list = get_dataset_info()

    # only solve the test set
    _, test_videonames = train_test_split()
    test_videonames_dict = {}
    for vn in test_videonames:
        test_videonames_dict[vn] = 1

    for (video_name, sym_axis, shape_xyz) in zip(video_names, sym_axis_list, shape_xyz_list):
        if not video_name in test_videonames_dict:
            continue

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

            use_completed_point_cloud = True
            completed_point_cloud = join(completed_point_cloud_dir, obj_category, sequence_name, "completed_point_cloud", "0000.ply")
            
            solve(sequence_name=sequence_name, sequence_dir=join(category_dir, sequence_name), save_dir=tracking_save_dir, kinematics_save_root=kinematics_save_dir, ids=ids, N_frames=5, use_completed_point_cloud=use_completed_point_cloud, completed_point_cloud=completed_point_cloud, sym_axis=sym_axis, shape_xyz=shape_xyz)
