import os
from os.path import join, dirname, abspath
import sys
BASEPATH = dirname(abspath(__file__))
sys.path.insert(0, join(BASEPATH, '..'))
import time
import numpy as np
import torch
from torch import nn
from optimizer.optimizer import Optimizer
from transforms3d.euler import euler2mat
from optimizer.torch_utils.torch_euler2mat import Euler2Mat
import open3d


class TorchModelGeometry(nn.Module):
    def __init__(self, visual_points, tactile_points, extrinsic, model_points, scale, init_state, device):
        super(TorchModelGeometry, self).__init__()
        # constants
        self.visual_points = nn.Parameter(torch.tensor(visual_points, dtype=torch.float32, device=device))
        self.visual_points.requires_grad = False
        self.tactile_points = nn.Parameter(torch.tensor(tactile_points, dtype=torch.float32, device=device))
        self.tactile_points.requires_grad = False
        self.extrinsic = nn.Parameter(torch.tensor(extrinsic, dtype=torch.float32, device=device))
        self.extrinsic.requires_grad = False
        self.model_points = nn.Parameter(torch.tensor(model_points, dtype=torch.float32, device=device))
        self.model_points.requires_grad = False
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32, device=device))
        self.scale.requires_grad = False
        self.init_state = torch.from_numpy(init_state).to(device)
        self.init_state.requires_grad = False

        # variables
        self.state = nn.Parameter(torch.tensor(init_state, dtype=torch.float32, device=device))
        self.state.requires_grad = True

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
    
    def forward(self):
        t = self.state[0:3].reshape(3)
        r_omega = self.state[3:6].reshape(3)
        r = Euler2Mat.apply(r_omega)
        
        E_cd = 0
        if self.visual_points.shape[0] > 0:
            vc_points = self.visual_points - t.reshape(1, 3)
            vc_points = torch.matmul(vc_points, r)
            vc_points = vc_points / self.scale
            E_cd += 1.0 * self.CDLoss(vc_points, self.model_points)

        if self.tactile_points.shape[0] > 0:
            tc_points = self.tactile_points - t.reshape(1, 3)
            tc_points = torch.matmul(tc_points, r)
            tc_points = tc_points / self.scale
            E_cd += 0.1 * self.CDLoss(tc_points, self.model_points)

        E = E_cd
        return E
    
    def get_state(self):
        return self.state.detach().cpu().numpy()


class TorchOptimizerGeometry(Optimizer):
    def __init__(self, visual_points, tactile_points, extrinsic, model_points, scale, init_state, device):
        super(TorchOptimizerGeometry, self).__init__()
        # model
        self.model = TorchModelGeometry(visual_points, tactile_points, extrinsic, model_points, scale, init_state, device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # answer
        self.answer = init_state
        self.answer_energy = self.model().item()
    
    def solve(self, epoch):
        for _ in range(epoch):
            self.optimizer.zero_grad()
            E = self.model()
            E.backward()
            self.optimizer.step()
        
        self.answer = self.model.get_state()
        self.answer_energy = self.model().item()

    def get_answer(self):
        return self.answer, self.answer_energy


# Unit Test
if __name__ == "__main__":
    optimizer = TorchOptimizerGeometry(np.zeros((1000, 3)), np.zeros((100, 3)), np.random.normal(0, 1, (4, 4)), np.ones((10000, 3)), scale=1, init_state=np.array([0, 0, 0, 1, 2, 3]), device="cuda:0")

    r1 = euler2mat(0.6, 1.6, -3)
    r2 = Euler2Mat.apply(torch.tensor([0.6, 1.6, -3], device="cuda:0")).detach().cpu().numpy()
    print(r1, r2)

    print(optimizer.get_answer())
    start_time = time.time()
    optimizer.solve(epoch=1000)
    print("time =", time.time() - start_time)
    print(optimizer.get_answer())
