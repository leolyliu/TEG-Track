import os
import numpy as np
import cv2
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
from get_data import read_intrinsics_from_txt, read_extrinsics_from_txt


def draw_bbox(img, x, y, color=(0, 255, 0)):
    p = np.vstack((x, y)).T.astype(np.int32)
    for j in range(8):
        cv2.circle(img, center=tuple(p[j]), radius=2, color=color)
    cv2.line(img, tuple(p[0]), tuple(p[1]), color, 2)
    cv2.line(img, tuple(p[0]), tuple(p[2]), color, 2)
    cv2.line(img, tuple(p[0]), tuple(p[3]), color, 2)
    cv2.line(img, tuple(p[1]), tuple(p[6]), color, 2)
    cv2.line(img, tuple(p[1]), tuple(p[7]), color, 2)
    cv2.line(img, tuple(p[2]), tuple(p[5]), color, 2)
    cv2.line(img, tuple(p[2]), tuple(p[7]), color, 2)
    cv2.line(img, tuple(p[3]), tuple(p[5]), color, 2)
    cv2.line(img, tuple(p[3]), tuple(p[6]), color, 2)
    cv2.line(img, tuple(p[4]), tuple(p[5]), color, 2)
    cv2.line(img, tuple(p[4]), tuple(p[6]), color, 2)
    cv2.line(img, tuple(p[4]), tuple(p[7]), color, 2)


def check_tracking_results(pred_poses, gt_poses, gt_corners, video_file, output_folder, frame_size=(1280, 720), scaling=True, start_idx=0, view_mode="egocentric", fps=30):

    assert view_mode == "3rd_view"

    gt_corner_pts = gt_corners
    N = len(pred_poses)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(os.path.join(output_folder, "tracking_visualization.mp4"), fourcc=fourcc, fps=fps, frameSize=frame_size)
    for i in range(N):
        ii = i
        intrinsic = read_intrinsics_from_txt(os.path.join(video_file, "camera_params", "visual_camera_intrinsic.txt"))
        extrinsic = read_extrinsics_from_txt(os.path.join(video_file, "camera_params", "visual_camera_extrinsic.txt"))
        img = cv2.imread(os.path.join(video_file, "rgb", str(start_idx + ii).zfill(4) + ".png"))
        
        if gt_poses is not None:
            corner_pts_world = gt_corner_pts.copy()
            if scaling:
                corner_pts_world = ((gt_poses[i]["rotation"] @ (gt_poses[i]["scale"] * corner_pts_world.T)) +
                                    gt_poses[i]["translation"]).T
            else:
                corner_pts_world = ((gt_poses[i]["rotation"] @ corner_pts_world.T) +
                                    gt_poses[i]["translation"]).T
            corner_pts_camera = ((extrinsic[:3, :3] @ corner_pts_world.T) + extrinsic[:3, 3:]).T
            corner_pts_image = (intrinsic @ corner_pts_camera.T).T
            corner_x_image = corner_pts_image[..., 0] / corner_pts_image[..., 2]
            corner_y_image = corner_pts_image[..., 1] / corner_pts_image[..., 2]
            draw_bbox(img, corner_x_image, corner_y_image, (0, 255, 0))

        corner_pts_world = gt_corner_pts.copy()
        if scaling:
            corner_pts_world = ((pred_poses[i]["rotation"] @ (pred_poses[i]["scale"] * corner_pts_world.T)) +
                                pred_poses[i]["translation"].reshape(3, 1)).T
        else:
            corner_pts_world = ((pred_poses[i]["rotation"] @ corner_pts_world.T) +
                                pred_poses[i]["translation"].reshape(3, 1)).T
        corner_pts_camera = ((extrinsic[:3, :3] @ corner_pts_world.T) + extrinsic[:3, 3:]).T
        corner_pts_image = (intrinsic @ corner_pts_camera.T).T
        corner_x_image = corner_pts_image[..., 0] / corner_pts_image[..., 2]
        corner_y_image = corner_pts_image[..., 1] / corner_pts_image[..., 2]
        draw_bbox(img, corner_x_image, corner_y_image, (0, 0, 255))
        os.makedirs(os.path.join(output_folder, 'bbox_img'), exist_ok=True)
        cv2.imwrite(os.path.join(output_folder, 'bbox_img', str(start_idx + i) + ".png"), img)
        videoWriter.write(img)
    videoWriter.release()
