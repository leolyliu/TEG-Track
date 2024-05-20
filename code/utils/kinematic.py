from gettext import translation
import numpy as np
from transforms3d.euler import mat2euler, euler2axangle, euler2mat
from utils.pose import get_pose_from_state


def get_S_from_omega(angular_velocity):
    S = np.zeros((3, 3))
    S[0, 1] = - angular_velocity[2]
    S[0, 2] = angular_velocity[1]
    S[1, 0] = angular_velocity[2]
    S[1, 2] = - angular_velocity[0]
    S[2, 0] = - angular_velocity[1]
    S[2, 1] = angular_velocity[0]
    return S


def get_omega_from_S(S):
    omega = np.zeros((3, 1))
    omega[0] = S[2, 1]
    omega[1] = S[0, 2]
    omega[2] = S[1, 0]
    return omega


def pred_rotation_from_omega(r0, omega, dt):
    theta = np.linalg.norm(omega)
    ax = omega / (theta + 1e-7)
    I = np.diag([1, 1, 1])
    S = get_S_from_omega(ax)
    r = (I + np.sin(theta * dt) * S + (1 - np.cos(theta * dt)) * (S @ S)) @ r0
    return r


def pred_velocity_from_pose(t, t0, r, r0, dt):
    v = ((t - t0) / dt).reshape(3, 1)
    euler_delta = mat2euler(r @ np.linalg.inv(r0))
    ax, theta = euler2axangle(euler_delta[0], euler_delta[1], euler_delta[2])
    theta /= dt
    omega = theta * ax
    S = get_S_from_omega(omega)
    omega = omega.reshape(3, 1)
    return v, S, omega


def pred_pose_mean_velocities(last_pose, v, omega, dt):
    t0 = last_pose["translation"]
    r0 = last_pose["rotation"]
    v = v.reshape(3, 1)
    omega = omega.reshape(3, 1)
    pred_translation = t0.reshape(3, 1) + v * dt
    pred_rotation = pred_rotation_from_omega(r0, omega, dt)
    pred_pose = {"rotation": pred_rotation, "translation": pred_translation}
    if "scale" in last_pose:
        pred_pose["scale"] = last_pose["scale"]
    return pred_pose


def pred_pose(last_pose, last_state, state, dt):
    t0 = last_pose["translation"]
    r0 = last_pose["rotation"]
    v = (state[-6:-3].reshape(3, 1) + last_state[-6:-3].reshape(3, 1)) * 0.5
    omega = (state[-3:].reshape(3, 1) + last_state[-3:].reshape(3, 1)) * 0.5
    pred_translation = t0.reshape(3, 1) + v * dt  # assume the object performs uniformly accelerated motion during the two adjacent frames
    pred_rotation = pred_rotation_from_omega(r0, omega, dt)  # assume the object performs uniformly accelerated motion during the two adjacent frames
    pred_pose = {"rotation": pred_rotation, "translation": pred_translation}
    if "scale" in last_pose:
        pred_pose["scale"] = last_pose["scale"]
    return pred_pose


def pred_poses_from_kinematics(states, scale=None, dt=1/30):
    assert len(states) > 0
    pred_poses = [get_pose_from_state(states[0], scale)]
    for i in range(1, len(states)):
        state = states[i]
        last_state = states[i - 1]
        last_pose = pred_poses[i - 1]
        if i == 1:
            last_state = state
        pose = pred_pose(last_pose, last_state, state, dt)
        pred_poses.append(pose)
    return pred_poses


def get_velocity_diff(state0, state1):
    v0 = state0[6:9]
    av0 = state0[9:12]
    v1 = state1[6:9]
    av1 = state1[9:12]
    return np.linalg.norm(v1 - v0), np.linalg.norm(av1 - av0)


def velocity_gripper2world(r, velocity, omega, pad_velocity):
    velocity_g = velocity + pad_velocity
    omega_g = r @ omega
    return velocity_g, omega_g


def velocity_world2gripper(r, velocity, omega, pad_velocity):
    velocity_w = velocity - pad_velocity
    omega_w = r.T @ omega
    return velocity_w, omega_w


def pose_gripper2world(r0, t0, pad_pose):
    r = pad_pose[:3, :3] @ r0
    t = (pad_pose[:3, :3] @ t0) + pad_pose[:3, 3:]
    return r, t


def pose_world2gripper(r0, t0, pad_pose):
    r = pad_pose[:3, :3].T @ r0
    t = pad_pose[:3, :3].T @ (t0 - pad_pose[:3, 3:])
    return r, t


def state_world2gripper(state0, pad_pose, pad_velocity):
    t0 = state0[:3].reshape(3, 1)
    r0 = euler2mat(state0[3], state0[4], state0[5])
    v0 = state0[6:9].reshape(3, 1)
    omega0 = state0[9:].reshape(3, 1)
    r, t = pose_world2gripper(r0, t0, pad_pose)
    v, omega = velocity_world2gripper(r0, v0, omega0, pad_velocity)
    state = np.zeros((12, 1))
    state[:3] = t.reshape(3, 1)
    ai, aj, ak = mat2euler(r)
    state[3:6] = np.array((ai, aj, ak)).reshape(3, 1)
    state[6:9] = v.reshape(3, 1)
    state[9:] = omega.reshape(3, 1)
    return r, t, v, omega, state
