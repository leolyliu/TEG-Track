import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from utils.bbox_vis import check_tracking_results


def get_corners_from_shapexyz(shape_xyz):
    x, y, z = list(shape_xyz / 2)
    c = np.float32([
        [-x, -y, -z],
        [x, -y, -z],
        [-x, y, -z],
        [-x, -y, z],
        [x, y, z],
        [-x, y, z],
        [x, -y, z],
        [x, y, -z]
    ])
    return c


def compute_lims_for_pts(all_points):
    corners = np.stack([np.min(all_points, axis=0), np.max(all_points, axis=0)], axis=0)
    center = corners.mean(axis=0)  # [1, 3]
    max_size = np.max(corners[1] - corners[0]) * 0.4
    lims = np.stack([center - max_size, center + max_size], axis=0).swapaxes(0, 1)
    return lims


def set_axes_equal(ax, limits=None, labels=True):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    ax.set_box_aspect([1, 1, 1])
    if limits is None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    else:
        x_limits, y_limits, z_limits = limits
        ax.set_xlim3d([x_limits[0], x_limits[1]])
        ax.set_ylim3d([y_limits[0], y_limits[1]])
        ax.set_zlim3d([z_limits[0], z_limits[1]])

    if labels:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')


def bbox_visualization(model_path, output_folder, pred_poses, gt_poses, path, frame_size=(1280, 720), scaling=True, start_idx=0, view_mode='3rd_view', fps=30, shape_xyz=None):
    if shape_xyz is None:
        model = o3d.io.read_triangle_mesh(model_path)
        bbox = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(model)
        gt_corners = np.asarray(bbox.get_box_points())
    else:
        gt_corners = get_corners_from_shapexyz(shape_xyz)
    check_tracking_results(pred_poses, gt_poses, gt_corners, path, output_folder, frame_size=frame_size, scaling=scaling, start_idx=start_idx, view_mode=view_mode, fps=fps)


def plot_bbox_with_pts(points, bcm, view_angle=None, lims=None,
                       dpi=100, s=2, save_path=None, show_fig=False, save_fig=False):

    fig = plt.figure(dpi=dpi, figsize=(6, 6))
    if lims is None:
        lims = compute_lims_for_pts(points)

    ax = plt.subplot(1, 1, 1, projection='3d', proj_type='ortho')
    if view_angle == None:
        ax.view_init(elev=0, azim=180)
    else:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    colors = ['dodgerblue', 'gold', 'mediumorchid', 'silver']

    # draw point cloud
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker='o', s=s,  c=colors[0])

    # draw bbox points
    ax.scatter(bcm[:, 0], bcm[:, 1], bcm[:, 2], marker='o', s=10, c=colors[2])

    # color_list = ['red', 'yellow', 'blue', 'green']
    color_s = 'lime'
    lw_s = 2

    for pair in [[0, 1], [0, 2], [0, 3], [1, 6], [1, 7], [2, 5], [2, 7], [3, 5], [3, 6], [4, 5], [4, 6], [4, 7]]:
        ax.plot3D([bcm[pair[0]][0], bcm[pair[1]][0]],
                  [bcm[pair[0]][1], bcm[pair[1]][1]],
                  [bcm[pair[0]][2], bcm[pair[1]][2]], color=color_s, linewidth=lw_s)
    plt.axis('off')
    plt.grid('off')
    set_axes_equal(ax, lims)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_fig:
        fig.savefig(save_path)

    plt.close()
    return lims


def bbox_visualization_pts(point_list, pred_poses, model_path, view_angle, result_path, fps=30, scaling=False):
    model = o3d.io.read_triangle_mesh(model_path)
    bbox = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(model)
    gt_corners = np.asarray(bbox.get_box_points())

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(os.path.join(result_path, "bbox_visualization_pts.mp4"), fourcc=fourcc, fps=fps, frameSize=(600, 600))
    save_dir = os.path.join(result_path, 'bbox_pts_img')
    os.makedirs(save_dir, exist_ok=True)
    for idx, points in enumerate(point_list):
        corner_pts_world = gt_corners.copy()
        pred_pose = pred_poses[idx]
        if scaling:
            bcm = ((pred_pose["rotation"] @ (pred_pose["scale"] * corner_pts_world.T)) + pred_pose["translation"]).T
        else:
            bcm = ((pred_pose["rotation"] @ corner_pts_world.T) + pred_pose["translation"].reshape(3, 1)).T  # bcm: [8, 3] bbox
        save_path = os.path.join(save_dir, str(idx) + '.png')
        plot_bbox_with_pts(points, bcm, view_angle, save_path=save_path, save_fig=True)
        img = cv2.imread(save_path)
        videoWriter.write(img)


def posediff_visualization(x, tdiffs=None, rdiffs=None, tdiffs_cam=None, rdiffs_cam=None, tdiffs_0=None, rdiffs_0=None, tdiffs_1=None, rdiffs_1=None, tdiffs_2=None, rdiffs_2=None, tdiffs_first=None, rdiffs_first=None, tdiffs_k=None, rdiffs_k=None, title="", save_path=""):
    plt.clf()
    if not tdiffs is None:
        plt.plot(x, tdiffs, color="red", label="tdiff_consecutive(cm)", linestyle="dashed")
        plt.plot(x, rdiffs, color="red", label="rdiff_consecutive(deg)")
    if not tdiffs_cam is None:
        plt.plot(x, tdiffs_cam, color="orange", label="tdiff_consec_cam(cm)", linestyle="dashed")
        plt.plot(x, rdiffs_cam, color="orange", label="rdiff_consec_cam(deg)")
    if not tdiffs_0 is None:
        plt.plot(x, tdiffs_0, color="blue", label="tdiff_gt_v(cm)", linestyle="dashed")
        plt.plot(x, rdiffs_0, color="blue", label="rdiff_gt_omega(deg)")
    if not tdiffs_1 is None:
        plt.plot(x, tdiffs_1, color="green", label="tdiff_optimize_single(cm)", linestyle="dashed")
        plt.plot(x, rdiffs_1, color="green", label="rdiff_optimize_single(deg)")
    if not tdiffs_2 is None:
        plt.plot(x, tdiffs_2, color="black", label="tdiff_optimize(cm)", linestyle="dashed")
        plt.plot(x, rdiffs_2, color="black", label="rdiff_optimize(deg)")
    if not tdiffs_first is None:
        plt.plot(x, tdiffs_first, color="purple", label="tdiff_optimize_first_result(cm)", linestyle="dashed")
        plt.plot(x, rdiffs_first, color="purple", label="rdiff_optimize_first_result(deg)")
    if not tdiffs_k is None:
        plt.plot(x, tdiffs_k, color="grey", label="tdiff_optimize_kinematics_only(cm)", linestyle="dashed")
        plt.plot(x, rdiffs_k, color="grey", label="rdiff_optimize_kinematics_only(deg)")
    plt.legend()
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("")
    plt.savefig(save_path)


def velocitydiff_visualization(x, v_diffs=None, av_diffs=None, v_diffs_first=None, av_diffs_first=None, title="", save_path=""):
    plt.clf()
    if not v_diffs is None:
        plt.plot(x, v_diffs, color="black", label="vdiff_optimize_result", linestyle="dashed")
        plt.plot(x, av_diffs, color="black", label="avdiff_optimize_result")
    if not v_diffs_first is None:
        plt.plot(x, v_diffs_first, color="purple", label="vdiff_optimize_first_result", linestyle="dashed")
        plt.plot(x, av_diffs_first, color="purple", label="avdiff_optimize_first_result")
    plt.legend()
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("")
    plt.savefig(save_path)


def velocity_temporal_diff_visualization(x, vs_gt=None, avs_gt=None, vs=None, avs=None, vs_first=None, avs_first=None, title="", save_path=""):
    plt.clf()
    x = np.array(x)[1:]
    if not vs_gt is None:
        vs_gt = np.array(vs_gt)
        avs_gt = np.array(avs_gt)
        vs_diff = np.linalg.norm(vs_gt[1:] - vs_gt[:-1], axis=1)
        avs_diff = np.linalg.norm(avs_gt[1:] - avs_gt[:-1], axis=1)
        plt.plot(x, vs_diff, color="red", label="v_temporal_diff_gt", linestyle="dashed")
        plt.plot(x, avs_diff, color="red", label="av_temporal_diff_gt")
    if not vs is None:
        vs = np.array(vs)
        avs = np.array(avs)
        vs_diff = np.linalg.norm(vs[1:] - vs[:-1], axis=1)
        avs_diff = np.linalg.norm(avs[1:] - avs[:-1], axis=1)
        plt.plot(x, vs_diff, color="black", label="v_temporal_diff_opt", linestyle="dashed")
        plt.plot(x, avs_diff, color="black", label="av_temporal_diff_opt")
    plt.legend()
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("")
    plt.savefig(save_path)


def physics_constant_diff_visualization(x, mass_diffs, inertia_diffs, title="", save_path=""):
    plt.clf()
    plt.plot(x, mass_diffs, color="red", label="mass_diffs")
    plt.plot(x, np.array(inertia_diffs) * 1000, color="blue", label="inertia_diffs * 1000")
    plt.legend()
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("")
    plt.savefig(save_path)


def motiondiff_visualization(x, vdiffs=None, omegadiffs=None, title="", save_path=""):
    plt.clf()
    

def energy_visualization(x, opt_energies, gt_energies, title, save_path):
    plt.clf()
    if "tsdf" in opt_energies[0]:
        opt_tsdf_energies = [e["tsdf"] for e in opt_energies]
        gt_tsdf_energies = [e["tsdf"] for e in gt_energies]
        plt.plot(x, opt_tsdf_energies, color="red", label="opt_E_tsdf")
        plt.plot(x, gt_tsdf_energies, color="red", label="gt_E_tsdf", linestyle="dashed")
    if "t" in opt_energies[0]:
        opt_t_energies = [e["t"] for e in opt_energies]
        gt_t_energies = [e["t"] for e in gt_energies]
        plt.plot(x, opt_t_energies, color="blue", label="opt_E_t")
        plt.plot(x, gt_t_energies, color="blue", label="gt_E_t", linestyle="dashed")
    if "r" in opt_energies[0]:
        opt_r_energies = [e["r"] for e in opt_energies]
        gt_r_energies = [e["r"] for e in gt_energies]
        plt.plot(x, opt_r_energies, color="green", label="opt_E_r")
        plt.plot(x, gt_r_energies, color="green", label="gt_E_r", linestyle="dashed")
    if "force" in opt_energies[0]:
        opt_force_energies = [e["force"] for e in opt_energies]
        gt_force_energies = [e["force"] for e in gt_energies]
        plt.plot(x, opt_force_energies, color="black", label="opt_E_force")
        plt.plot(x, gt_force_energies, color="black", label="gt_E_force", linestyle="dashed")
    if "torque" in opt_energies[0]:
        opt_torque_energies = [e["torque"] for e in opt_energies]
        gt_torque_energies = [e["torque"] for e in gt_energies]
        plt.plot(x, opt_torque_energies, color="grey", label="opt_E_torque")
        plt.plot(x, gt_torque_energies, color="grey", label="gt_E_torque", linestyle="dashed")
    if "contact" in opt_energies[0]:
        opt_contact_energies = [e["contact"] for e in opt_energies]
        gt_contact_energies = [e["contact"] for e in gt_energies]
        plt.plot(x, opt_contact_energies, color="brown", label="opt_E_contact")
        plt.plot(x, gt_contact_energies, color="brown", label="gt_E_contact", linestyle="dashed")
    if "sum" in opt_energies[0]:
        opt_sum_energies = [e["sum"] for e in opt_energies]
        gt_sum_energies = [e["sum"] for e in gt_energies]
        plt.plot(x, opt_sum_energies, color="purple", label="opt_E_sum")
        plt.plot(x, gt_sum_energies, color="purple", label="gt_E_sum", linestyle="dashed")
    plt.legend()
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("")
    plt.savefig(save_path)


def pcd_visualization(all_points, pose_gt, pose_opt, scale, path, frame_idx):
    ps_gt = np.matmul(all_points - pose_gt["translation"].reshape(1, 3), pose_gt["rotation"]) / scale
    ps_opt = np.matmul(all_points - pose_opt["translation"].reshape(1, 3), pose_opt["rotation"]) / scale
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ps_gt)
    o3d.io.write_point_cloud(os.path.join(path, str(frame_idx) + "_gt.ply"), pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ps_opt)
    o3d.io.write_point_cloud(os.path.join(path, str(frame_idx) + "_opt.ply"), pcd)
