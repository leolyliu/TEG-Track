import os
from os.path import join
import argparse
import numpy as np
import pickle
import json
import cv2
import open3d as o3d


def read_metadata(file_path):
    metadata = pickle.load(open(file_path, "rb"))
    return metadata


def read_sequence_names(file_path):
    sequence_names = json.load(open(file_path, "r"))
    return sequence_names


def read_visual_sensor_intrinsic(file_path):
    intrinsic = np.loadtxt(file_path)
    return intrinsic


def read_visual_sensor_extrinsic(file_path):
    extrinsic = np.loadtxt(file_path)
    return extrinsic


def read_visual_sensor_rgb(file_path):
    rgb = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.uint8)  # shape = (720, 1280, 3), channels = RGB
    return rgb


def read_visual_sensor_depth(file_path):
    depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    H, W = depth.shape[:2]
    depth = np.float32(depth).reshape((H, W)) / 1000  # shape = (720, 1280), unit: m
    return depth


def read_tactile_sensor_rgb(file_path):
    rgb = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.uint8)  # shape = (480, 640, 3), channels = RGB
    return rgb


def read_tactile_sensor_depth(file_path):
    depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    H, W = depth.shape[:2]
    depth = np.float32(depth).reshape((H, W)) / 100000  # shape = (480, 640), unit: m
    return depth


def read_tactile_sensor_point_cloud(file_path):
    object_tactile_point_cloud = o3d.io.read_point_cloud(file_path)
    return object_tactile_point_cloud


def read_tactile_sensor_pad_pose(file_path):
    pad2world = np.loadtxt(file_path)
    return pad2world


def read_tactile_sensor_marker_pixels(file_path):
    pixels = np.loadtxt(file_path)  # shape = (2, N_marker)
    return pixels


def read_object_pose(file_path):
    """
    return: a 4x4 transformation matrix, indicating the pose of the object relative to the world.
    """
    object2world = np.load(file_path)  # shape = (4, 4)
    return object2world


def read_object_mask(file_path):
    object_mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.uint8)  # shape = (720, 1280, 3), channels = RGB
    object_mask = object_mask[:, :, 0] > 0  # shape = (720, 1280)
    return object_mask


def read_object_point_cloud(file_path):
    object_point_cloud = o3d.io.read_point_cloud(file_path)
    return object_point_cloud


def read_object_contact_information(file_path):
    object_contact_information = pickle.load(open(file_path, "rb"))
    for key in object_contact_information:
        object_contact_information[key] = np.float32(object_contact_information[key])
    return object_contact_information


def visualize_one_sequence(data_dir, save_dir):
    
    N_frame = 401

    visual_camera_intrinsic = read_visual_sensor_intrinsic(join(data_dir, "camera_params", "visual_camera_intrinsic.txt"))
    visual_camera_extrinsic = read_visual_sensor_intrinsic(join(data_dir, "camera_params", "visual_camera_extrinsic.txt"))

    # save videos
    os.makedirs(save_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(join(save_dir, "visual_sensor_rgb.mp4"), fourcc, 30, (1280, 1200))
    
    for frame_idx in range(N_frame):
        
        frame_name = str(frame_idx).zfill(4)

        # visual sensor signals
        rgb = read_visual_sensor_rgb(join(data_dir, "rgb", frame_name + ".png"))
        depth = read_visual_sensor_depth(join(data_dir, "depth", frame_name + ".png"))
        mask = read_object_mask(join(data_dir, "object_mask", frame_name + ".png"))
        object_visual_point_cloud = read_object_point_cloud(join(data_dir, "visual_point_cloud", frame_name + ".ply"))

        # left tactile sensor signals
        left_rgb = read_tactile_sensor_rgb(join(data_dir, "left_rgb", frame_name + ".png"))
        left_depth = read_tactile_sensor_depth(join(data_dir, "left_depth", frame_name + ".png"))
        left_point_cloud = read_tactile_sensor_point_cloud(join(data_dir, "left_point_cloud", frame_name + ".ply"))
        leftpad_to_world = read_tactile_sensor_pad_pose(join(data_dir, "left_pad_pose", frame_name + ".txt"))
        left_marker_pixels = read_tactile_sensor_marker_pixels(join(data_dir, "left_marker_point_positions", frame_name + ".txt"))

        # right tactile sensor signals
        right_rgb = read_tactile_sensor_rgb(join(data_dir, "right_rgb", frame_name + ".png"))
        right_depth = read_tactile_sensor_depth(join(data_dir, "right_depth", frame_name + ".png"))
        right_point_cloud = read_tactile_sensor_point_cloud(join(data_dir, "right_point_cloud", frame_name + ".ply"))
        rightpad_to_world = read_tactile_sensor_pad_pose(join(data_dir, "right_pad_pose", frame_name + ".txt"))
        right_marker_pixels = read_tactile_sensor_marker_pixels(join(data_dir, "right_marker_point_positions", frame_name + ".txt"))

        # object pose
        object_pose = read_object_pose(join(data_dir, "object_pose", frame_name + ".npy"))

        # precomputed object contact information
        if frame_idx > 0:
            object_contact_information = read_object_contact_information(join(data_dir, "precomputed_object_contact_information", frame_name + ".pkl"))

        # visualization
        img = np.zeros((1200, 1280, 3)).astype(np.uint8)
        img[:720, :] = rgb
        img[720:, :640] = left_rgb
        img[720:, 640:] = right_rgb
        img = img[:, :, ::-1].astype(np.uint8)  # RGB to BGR
        video_writer.write(img)

    video_writer.release()


def get_args():
    parser = argparse.ArgumentParser()
    ###########################################################
    parser.add_argument('--dataset_dir', type=str)
    ###########################################################

    args = parser.parse_args()
    return args



if __name__ == "__main__":

    ###########################################################
    args = get_args()
    dataset_dir = args.dataset_dir
    save_root_dir = "./"
    category = "camera"
    sequence_name = "0907_camera2_1_000"
    ###########################################################

    data_dir = join(dataset_dir, category, sequence_name)
    save_dir = join(save_root_dir, category, sequence_name)

    visualize_one_sequence(data_dir=data_dir, save_dir=save_dir)
