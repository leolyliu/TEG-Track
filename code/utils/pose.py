import numpy as np
import copy
from transforms3d.euler import euler2mat, mat2euler, euler2quat, quat2mat


def pose_dict2mat(pose):
    T = np.eye(4)
    T[:3, :3] = pose["rotation"]
    T[:3, 3:] = pose["translation"].reshape(3, 1)
    return T


def pose_torch2np(pose):
    pose["rotation"] = pose["rotation"].detach().cpu().numpy()
    pose["translation"] = pose["translation"].detach().cpu().numpy()
    pose["scale"] = pose["scale"].detach().cpu().numpy()
    return pose


def proj_u_a(u, a):
    """
    Used for Schmidt Orthogonalization
    """
    top = np.sum(u * a)
    bottom = max(np.sum(u * u), 1e-8)
    return (top / bottom) * u


def compute_rotation_matrix_from_matrix(M):  # numpy, shape = (3, 3)
    """
    Schmidt Orthogonalization
    """
    a1 = M[:, 0]
    a2 = M[:, 1]
    a3 = M[:, 2]
    u1 = a1
    u2 = a2 - proj_u_a(u1, a2)
    u3 = a3 - proj_u_a(u1, a3) - proj_u_a(u2, a3)
    u1 /= max(np.linalg.norm(u1), 1e-8)
    u2 /= max(np.linalg.norm(u2), 1e-8)
    u3 /= max(np.linalg.norm(u3), 1e-8)
    return np.concatenate((u1.reshape(3, 1), u2.reshape(3, 1), u3.reshape(3, 1)), axis=1)


def pose_captra2opt(pose_captra):
    r = pose_captra["rotation"].detach().cpu().numpy().reshape(3, 3)
    t = pose_captra["translation"].detach().cpu().numpy().reshape(3)
    s = pose_captra["scale"].detach().cpu().numpy().reshape(1)[0]
    r = compute_rotation_matrix_from_matrix(r)
    return {"rotation": r, "translation": t, "scale": s}


def pose_world2cam(r0, t0, extrinsic):
    r = extrinsic[:3, :3] @ r0
    t = (extrinsic[:3, :3] @ t0) + extrinsic[:3, 3:]
    return r, t


def pose_cam2world(r0, t0, extrinsic):
    r = extrinsic[:3, :3].T @ r0
    t = extrinsic[:3, :3].T @ (t0 - extrinsic[:3, 3:])
    return r, t


def get_gripper_pose_from_world(world_pose, extrinsic, scale_factor=1.0):
    t_shape = world_pose["translation"].shape
    t = world_pose["translation"].reshape(3)
    r = world_pose["rotation"]
    s = world_pose["scale"]

    s = scale_factor * s
    r, t = pose_world2cam(r, t.reshape(3, 1), extrinsic)
    t = t.reshape(t_shape)
    gripper_pose = {"translation": t, "rotation": r, "scale": s}
    return gripper_pose


def get_world_pose_from_gripper(gripper_pose, extrinsic, scale_factor=1.0):
    t_shape = gripper_pose["translation"].shape
    t = gripper_pose["translation"].reshape(3)
    r = gripper_pose["rotation"]
    s = gripper_pose["scale"]

    s = s / scale_factor
    r, t = pose_cam2world(r, t.reshape(3, 1), extrinsic)
    t = t.reshape(t_shape)
    world_pose = {"translation": t, "rotation": r, "scale": s}
    return world_pose


def normalized_euler(a, b, c):
    a = (a + np.pi) % (2 * np.pi) - np.pi
    b = (b + np.pi) % (2 * np.pi) - np.pi
    c = (c + np.pi) % (2 * np.pi) - np.pi
    return a, b, c


def rot_diff_rad(r0, r1, sym_axis=None):
    if sym_axis == "x":
        x1, x2 = r0[..., 0], r1[..., 0]
        diff = np.sum(x1 * x2, axis=-1)
        diff = np.clip(diff, a_min=-1.0, a_max=1.0)
        return np.arccos(diff)
    elif sym_axis == "y":
        y1, y2 = r0[..., 1], r1[..., 1]
        diff = np.sum(y1 * y2, axis=-1)
        diff = np.clip(diff, a_min=-1.0, a_max=1.0)
        return np.arccos(diff)
    elif sym_axis == "z":
        z1, z2 = r0[..., 2], r1[..., 2]
        diff = np.sum(z1 * z2, axis=-1)
        diff = np.clip(diff, a_min=-1.0, a_max=1.0)
        return np.arccos(diff)
    else:
        mat_diff = np.matmul(r0, r1.swapaxes(-1, -2))
        diff = mat_diff[0, 0] + mat_diff[1, 1] + mat_diff[2, 2]
        diff = (diff - 1) / 2.0
        diff = np.clip(diff, a_min=-1.0, a_max=1.0)
        return np.arccos(diff)


def get_pose_diff(pose0, pose1, sym_axis=None):
    t0 = pose0["translation"].reshape(3, 1)
    r0 = pose0["rotation"]
    t1 = pose1["translation"].reshape(3, 1)
    r1 = pose1["rotation"]
    if ("scale" in pose0) and ("scale" in pose1):
        s0 = pose0["scale"]
        s1 = pose1["scale"]
        sdiff = np.abs(s1 - s0)
    else:
        sdiff = 0
    tdiff = np.linalg.norm(t1 - t0, ord=2)
    rdiff = rot_diff_rad(r0, r1, sym_axis=sym_axis) / np.pi * 180.0
    return sdiff, tdiff, rdiff


def get_pose_from_state(state, scale=None):
    t = state[0:3].reshape(3, 1)
    r = euler2mat(state[3], state[4], state[5])
    pose = {"translation": t, "rotation": r}
    if not scale is None:
        pose["scale"] = scale
    return pose


def get_poses_from_states(states, scale):
    poses = []
    for state in states:
        poses.append(get_pose_from_state(state, scale))
    return poses


def trans_bbox(bbox_info, pose):  # bbox_info.shape = (8, 3)
    bbox = bbox_info.T
    scale = 1.0
    if "scale" in pose:
        scale = pose["scale"]
    bbox = (pose["rotation"] @ (scale * bbox)) + pose["translation"].reshape(3, 1)
    bbox = bbox.T
    return bbox


def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)
    u1 = bbox[1] - bbox[0]
    u2 = bbox[2] - bbox[0]
    u3 = bbox[4] - bbox[0]

    up = pts - np.reshape(bbox[0], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1 > 0, p1 < np.dot(u1, u1))
    p2 = np.logical_and(p2 > 0, p2 < np.dot(u2, u2))
    p3 = np.logical_and(p3 > 0, p3 < np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)


def compute_IoU(pose0, pose1, bbox_info, K=10000):  # bbox_info.shape = (8, 3)
    bbox0 = trans_bbox(bbox_info, pose0)
    bbox1 = trans_bbox(bbox_info, pose1)
    bbox = np.concatenate((bbox0, bbox1), axis=0)
    bound_mn = np.min(bbox, axis=0)
    bound_mx = np.max(bbox, axis=0)
    rand_points = np.random.uniform(bound_mn, bound_mx, (K, 3))
    flag0 = pts_inside_box(rand_points, bbox0)
    flag1 = pts_inside_box(rand_points, bbox1)

    intersect = np.sum(np.logical_and(flag0, flag1))
    union = np.sum(np.logical_or(flag0, flag1))
    if union == 0:
        return 0.0
    else:
        return float(intersect) / float(union) * 100  # percentage


def compute_IoU_with_symaxis(pose0, pose1, bbox_info, sym_axis=None):  # bbox_info.shape = (8, 3)
    delta = 6
    N = int(np.ceil(360 / delta))

    poses = []
    if sym_axis is None:
        poses.append(pose0)
    elif sym_axis == "y":
        for i in range(N):
            R = euler2mat(0, (i * delta) / 180 * np.pi, 0)
            p = copy.deepcopy(pose0)
            p["rotation"] = p["rotation"] @ R
            poses.append(p)
    else:
        raise NotImplementedError
    
    mx_IoU = 0
    for i in range(len(poses)):
        IoU = compute_IoU(poses[i], pose1, bbox_info)
        mx_IoU = max(mx_IoU, IoU)
    return mx_IoU


def rotation_lerp(R0, R1, t):
    """
    R0: shape = (3, 3)
    R1: shape = (3, 3)
    t: 0 <= t <= 1, (1-t) * R0 + t * R1
    return: a rotation matrix, shape = (3, 3)
    """
    e0 = mat2euler(R0)
    q0 = euler2quat(e0[0], e0[1], e0[2])
    if q0[0] < 0:
        q0 = -q0
    e1 = mat2euler(R1)
    q1 = euler2quat(e1[0], e1[1], e1[2])
    if q1[0] < 0:
        q1 = -q1
    q = (1-t) * q0 + t * q1
    q /= max(np.linalg.norm(q), 1e-8)
    R = quat2mat(q)
    return R
