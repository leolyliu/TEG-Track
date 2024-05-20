import numpy as np


def solve_newton_equation(dt, forces, velocities):
    N_frames = len(forces)
    A = []
    b = []
    for frame_idx in range(1, N_frames - 1):
        v0 = velocities[frame_idx].reshape(3)
        v = velocities[frame_idx + 1].reshape(3)
        acceleration = (v - v0) / dt
        g = np.array((0, 0, -9.81))
        A = np.concatenate((A, acceleration - g))
        b = np.concatenate((b, forces[frame_idx]))
    A = A.reshape(A.shape[0], 1)
    b = b.reshape(b.shape[0], 1)
    m = np.linalg.lstsq(A, b, rcond=None)[0][0][0]
    return m


def solve_euler_equation(dt, torques, angular_velocities):
    N_frames = len(torques)
    A = np.empty((0, 3))
    b = []
    for frame_idx in range(1, N_frames - 1):
        omega0 = angular_velocities[frame_idx].reshape(3)
        omega = angular_velocities[frame_idx + 1].reshape(3)
        mean_omega = (omega0 + omega) / 2
        mean_angular_acceleration = (omega - omega0) / dt
        torque = torques[frame_idx].reshape(3)
        A = np.vstack((A, np.diag(mean_angular_acceleration)+np.cross(mean_omega, np.diag(mean_omega))))
        b = np.concatenate((b, torque))
    b = b.reshape(b.shape[0], 1)
    inertia = np.linalg.lstsq(A, b, rcond=None)[0].reshape(3)
    return inertia
