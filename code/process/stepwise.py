import numpy as np
import os
import sys
from os.path import join, dirname, abspath
BASEPATH = dirname(abspath(__file__))
sys.path.insert(0, join(BASEPATH, '..'))
from optimizer.torch_multigeo_optimizer import TorchOptimizerMultiGeo
from utils.kinematic import pred_rotation_from_omega
from transforms3d.euler import mat2euler, euler2mat


def optimize_multigeo(N_frames, model_points, delta_time, scale, rawdatas, pred_states, gt_states, device):
    init_state = pred_states[0][:6]  # inistalization

    visual_points_list = [data["visual_points"] for data in rawdatas]
    tactile_points_list = [data["tactile_points"] for data in rawdatas]
    extrinsics = [data["extrinsic"] for data in rawdatas]
    optimizer = TorchOptimizerMultiGeo(N_frames, pred_states[:N_frames], delta_time, visual_points_list, tactile_points_list, extrinsics, model_points, scale, init_state, device)
    optimizer.solve(epoch=50)
    state, opt_energy = optimizer.get_answer()
    print("multigeo_opt_energy:", opt_energy)

    if gt_states is not None:
        init_state = gt_states[0][:6]
        optimizer = TorchOptimizerMultiGeo(N_frames, gt_states[:N_frames], delta_time, visual_points_list, tactile_points_list, extrinsics, model_points, scale, init_state, device)
        _, gt_energy = optimizer.get_answer()
        print("multigeo_gt_energy:", gt_energy)

    t = state[0:3]
    r = euler2mat(state[3], state[4], state[5])
    states = []
    for i in range(N_frames):
        dt = delta_time[i]
        v = pred_states[i][6:9].reshape(3)
        omega = pred_states[i][9:].reshape(3)
        if not i == 0:
            t += v * dt
            r = pred_rotation_from_omega(r, omega, dt)
        ex, ey, ez = mat2euler(r)
        euler = np.array([ex, ey, ez])
        states.append(np.concatenate((t, euler, v, omega)))
    return states
