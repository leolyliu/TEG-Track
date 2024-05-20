import os
from os.path import join, dirname, abspath
import sys
BASEPATH = dirname(abspath(__file__))
sys.path.insert(0, join(BASEPATH, '..'))
import numpy as np
from utils.kinematic import get_S_from_omega


def cross(a, b):
    """
    cross product of (a, b)
    """

    assert a.shape == b.shape
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]).reshape(a.shape)


def solve_object_velocities(contact_points, contact_velocities, init_t, iter=10):
    N_contact = len(contact_points)
    v = None
    omega = None
    t = init_t

    A1 = np.zeros((3 * N_contact, 6))
    b1 = np.zeros((3 * N_contact, 1))
    for idx in range(N_contact):
        b1[3 * idx : 3 * idx + 3, :] = contact_velocities[idx].reshape(3, 1)
        A1[3 * idx : 3 * idx + 3, 0 : 3] = np.identity(3)
    A2 = np.zeros((3 * N_contact, 3))
    b2 = np.zeros((3 * N_contact, 1))

    # fix t, optimize v & omega
    A = A1
    b = b1
    for idx in range(N_contact):
        S = get_S_from_omega(t - contact_points[idx])
        A[3 * idx : 3 * idx + 3, 3 : 6] = S
        
    answer = np.linalg.lstsq(A, b, rcond=None)[0]
        
    v = answer[0:3].reshape(3)
    omega = answer[3:6].reshape(3)

    return v, omega, t
