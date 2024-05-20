import os
from os.path import join, dirname, abspath
import sys
BASEPATH = dirname(abspath(__file__))
sys.path.insert(0, join(BASEPATH, '..'))
import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from optimizer.optimizer import Optimizer
from transforms3d.euler import euler2mat, euler2quat, quat2mat
from optimizer.torch_utils.torch_euler2mat import Euler2Mat
from optimizer.torch_utils.torch_get_S_from_omega import GetSFromOmega


class TorchModelMultiGeo(nn.Module):
    def __init__(self, N_frame, physics_pred_states, delta_time, visual_points_list, tactile_points_list, extrinsics, model_points, scale, init_state, device):
        super(TorchModelMultiGeo, self).__init__()
        self.N_frame = N_frame
        self.delta_time = delta_time

        # category prior
        self.scale = scale

        # shape prior
        self.model_points = nn.Parameter(torch.tensor(model_points, dtype=torch.float32, device=device))
        self.model_points.requires_grad = False

        # data input
        self.extrinsics = nn.Parameter(torch.tensor(np.array(extrinsics), dtype=torch.float32, device=device))
        self.extrinsics.requires_grad = False
        self.visual_points_list = []
        for visual_points in visual_points_list:
            visual_points = nn.Parameter(torch.tensor(visual_points, dtype=torch.float32, device=device))
            visual_points.requires_grad = False
            self.visual_points_list.append(visual_points)
        self.tactile_points_list = []
        for tactile_points in tactile_points_list:
            tactile_points = nn.Parameter(torch.tensor(tactile_points, dtype=torch.float32, device=device))
            tactile_points.requires_grad = False
            self.tactile_points_list.append(tactile_points)

        # prediction from physics
        self.physics_pred_states = nn.Parameter(torch.tensor(np.array(physics_pred_states), dtype=torch.float32, device=device))
        self.physics_pred_states.requires_grad = False

        # variables
        self.state = nn.Parameter(torch.tensor(init_state, dtype=torch.float32, device=device))
        self.state.requires_grad = True

        # results
        self.E_dict = None

        # device
        self.device = device

    def CDLoss(self, pred, gt):
        # single-direction CD Loss
        a = torch.sum(pred * pred, dim=1).unsqueeze(dim=1)
        a = a.expand(pred.shape[0], gt.shape[0])
        b = torch.sum(gt * gt, dim=1).unsqueeze(dim=0)
        b = b.expand(pred.shape[0], gt.shape[0])
        c = torch.einsum('ik,jk->ij', pred, gt)
        dist = a + b - 2 * c
        cd_loss_1 = torch.mean(dist.min(dim=1)[0])
        return cd_loss_1

    def pred_rotation_from_omega(self, r0, omega, dt):
        theta = torch.norm(omega)
        ax = omega / torch.clip(theta, 1e-8, None)
        I = torch.diag(torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device))
        S = GetSFromOmega.apply(ax)
        r = torch.matmul(I + torch.sin(theta * dt) * S + (1 - torch.cos(theta * dt)) * torch.matmul(S, S), r0)
        return r

    def rot_diff_rad(self, r0, r1, sym_axis):
        if sym_axis == "x":
            x1, x2 = r0[..., 0], r1[..., 0]  # [3]
            diff = torch.dot(x1, x2)  # [1]
            diff = torch.clip(diff, -1.0 + 1e-6, 1.0 - 1e-6)
            return torch.arccos(diff)
        elif sym_axis == "y":
            y1, y2 = r0[..., 1], r1[..., 1]  # [3]
            diff = torch.dot(y1, y2)  # [1]
            diff = torch.clip(diff, -1.0 + 1e-6, 1.0 - 1e-6)
            return torch.arccos(diff)
        elif sym_axis == "z":
            z1, z2 = r0[..., 2], r1[..., 2]  # [3]
            diff = torch.dot(z1, z2)  # [1]
            diff = torch.clip(diff, -1.0 + 1e-6, 1.0 - 1e-6)
            return torch.arccos(diff)
        else:
            mat_diff = torch.matmul(r0, r1.T)
            diff = mat_diff[0, 0] + mat_diff[1, 1] + mat_diff[2, 2]
            diff = (diff - 1) / 2.0
            diff = torch.clip(diff, -1.0 + 1e-6, 1.0 - 1e-6)
            return torch.arccos(diff)

    def forward(self):
        E_cd = 0
        t0 = self.state[:3].reshape(3)
        r0_omega = self.state[3:].reshape(3)
        r0 = Euler2Mat.apply(r0_omega)
        t = t0
        r = r0
        for frame_idx in range(self.N_frame):
            if not frame_idx == 0:
                v = self.physics_pred_states[frame_idx][6:9].reshape(3)
                omega = self.physics_pred_states[frame_idx][9:].reshape(3)
                t = t + v * self.delta_time[frame_idx]
                r = self.pred_rotation_from_omega(r, omega, self.delta_time[frame_idx])
            if self.visual_points_list[frame_idx].shape[0] > 0:
                vc_points = self.visual_points_list[frame_idx] - t.reshape(1, 3)
                vc_points = torch.matmul(vc_points, r)
                vc_points = vc_points / self.scale
                E_cd += 1.0 * self.CDLoss(vc_points, self.model_points)
            if self.tactile_points_list[frame_idx].shape[0] > 0:
                tc_points = self.tactile_points_list[frame_idx] - t.reshape(1, 3)
                tc_points = torch.matmul(tc_points, r)
                tc_points = tc_points / self.scale
                E_cd += 0.1 * self.CDLoss(tc_points, self.model_points)
        return E_cd

    def set_state(self, state):
        self.state = nn.Parameter(torch.tensor(state, dtype=torch.float32, device=self.device))
        self.state.requires_grad = True

    def get_state(self):
        return self.state.detach().cpu().numpy()

    def get_E_dict(self):
        return self.E_dict


class TorchOptimizerMultiGeo(Optimizer):
    def __init__(self, N_frame, physics_pred_states, delta_time, visual_points_list, tactile_points_list, extrinsics, model_points, scale, init_state, device):
        super(TorchOptimizerMultiGeo, self).__init__()
        
        # model
        self.model = TorchModelMultiGeo(N_frame, physics_pred_states, delta_time, visual_points_list, tactile_points_list, extrinsics, model_points, scale, init_state, device)
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=50)
        
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        
        # answer
        self.answer = init_state
        self.answer_energy = self.model().item()

    def solve(self, epoch):
        '''
        # LBFGS
        def closure():
            E = self.model()
            self.optimizer.zero_grad()
            E.backward()
            return E
        self.model.train()
        self.optimizer.step(closure)
        '''
        state_list = []
        energy_list = []
        self.model.train()
        for _ in range(epoch):
            self.optimizer.zero_grad()
            E = self.model()
            state_list.append(self.model.get_state())
            energy_list.append(E.item())
            E.backward()
            self.optimizer.step()
            # self.scheduler.step()

        min_idx = np.argmin(np.array(energy_list))
        self.model.set_state(state_list[min_idx])
        self.answer = self.model.get_state()
        self.answer_energy = self.model().item()

    def get_answer(self):
        return self.answer, self.answer_energy

    def get_E_dict(self):
        return self.model.get_E_dict()
