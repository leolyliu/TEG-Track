import os
from os.path import join, isfile
import numpy as np
import cv2
import torch
from transforms3d.euler import mat2euler
from torch.utils.data import Dataset


class Tactile_Dataset(Dataset):
    def __init__(self, dataset_dir="", video_names=[], device="cpu", img_size=(640, 480), min_span=1, max_span=1, mode="train"):
        super(Tactile_Dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.video_names = video_names
        self.device = device
        self.img_size = img_size
        self.min_span = min_span
        self.max_span = max_span
        self.mode = mode

        assert (1 <= self.min_span) and (self.min_span <= self.max_span)

        self.prepare_data()
    
    def prepare_data(self):
        self.datapath = []
        self.a_valid_data = None
        for video_name in self.video_names:
            print("preparing: ", video_name)

            obj_name = video_name.split("_")[1]
            obj_category = obj_name[:-1]
            category_dir = join(self.dataset_dir, obj_category)
            sequence_list = []
            for sequence_name in os.listdir(category_dir):
                if sequence_name.startswith(video_name):
                    sequence_list.append(sequence_name)
            sequence_list.sort()
            for sequence_name in sequence_list:
                sequence_dir = join(category_dir, sequence_name)
                L_idx = 0
                while True:
                    R_idx = L_idx + self.min_span
                    if not isfile(join(sequence_dir, "rgb", str(R_idx).zfill(4) + ".png")):
                        break
                    
                    L_pose = np.load(join(sequence_dir, "object_pose", str(L_idx).zfill(4) + ".npy"))
                    R_pose = np.load(join(sequence_dir, "object_pose", str(R_idx).zfill(4) + ".npy"))
                    if (np.max(np.abs(L_pose)) > 100) or (np.max(np.abs(R_pose)) > 100):
                        L_idx += 1
                        continue
                    elif self.a_valid_data is None:
                        self.a_valid_data = [sequence_dir, L_idx, R_idx]

                    self.datapath.append([sequence_dir, L_idx, "left"])
                    self.datapath.append([sequence_dir, L_idx, "right"])
                    L_idx += 1

        self.len = len(self.datapath)
        print("finish dataset constructing, len(data) =", self.len)
    
    def __len__(self):
        return self.len
    
    def augmentation(self, L_idx, R_idx):
        if np.random.randint(0, 2) == 0:  # swap
            x = L_idx
            L_idx = R_idx
            R_idx = x
        return L_idx, R_idx
    
    def __getitem__(self, idx):
        sequence_dir, L_idx, sensor = self.datapath[idx]
        R_idx = L_idx + self.max_span
        while not isfile(join(sequence_dir, "rgb", str(R_idx).zfill(4) + ".png")):
            R_idx -= 1
        R_idx = np.random.randint(L_idx + self.min_span, R_idx + 1)

        # check data validity
        L_pose = np.load(join(sequence_dir, "object_pose", str(L_idx).zfill(4) + ".npy"))
        R_pose = np.load(join(sequence_dir, "object_pose", str(R_idx).zfill(4) + ".npy"))
        if (np.max(np.abs(L_pose)) > 100) or (np.max(np.abs(R_pose)) > 100):
            sequence_dir, L_idx, R_idx = self.a_valid_data
        
        L_idx, R_idx = self.augmentation(L_idx, R_idx)

        L_rgb = cv2.imread(join(sequence_dir, sensor + "_rgb", str(L_idx).zfill(4) + ".png"), cv2.IMREAD_UNCHANGED)
        L_model_matrix = np.loadtxt(join(sequence_dir, sensor + "_pad_pose", str(L_idx).zfill(4) + ".txt"))  # camera pose
        L_pose = np.load(join(sequence_dir, "object_pose", str(L_idx).zfill(4) + ".npy"))

        R_rgb = cv2.imread(join(sequence_dir, sensor + "_rgb", str(R_idx).zfill(4) + ".png"), cv2.IMREAD_UNCHANGED)
        R_model_matrix = np.loadtxt(join(sequence_dir, sensor + "_pad_pose", str(R_idx).zfill(4) + ".txt"))  # camera pose
        R_pose = np.load(join(sequence_dir, "object_pose", str(R_idx).zfill(4) + ".npy"))
        
        # resize
        L_rgb = cv2.resize(L_rgb, self.img_size)
        R_rgb = cv2.resize(R_rgb, self.img_size)
        delta_rgb = (R_rgb.astype(np.float32) - L_rgb.astype(np.float32)) / 255

        cam_L_pose = np.linalg.inv(L_model_matrix) @ L_pose  # object to camera
        cam_R_pose = np.linalg.inv(R_model_matrix) @ R_pose  # object to camera
        delta_pose = cam_R_pose @ np.linalg.inv(cam_L_pose)  # object to camera
        ai, aj, ak = mat2euler(delta_pose[:3, :3])

        return {
            "rgb": torch.as_tensor(delta_rgb.transpose(2, 0, 1)).to(self.device),
            "R": torch.as_tensor(np.float32([ai, aj, ak]) * 5).to(self.device),
            "t": torch.as_tensor(np.float32(delta_pose[:3, 3]) * 100).to(self.device),
        }
