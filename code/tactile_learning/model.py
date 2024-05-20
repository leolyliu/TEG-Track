import torch
from torch import nn
import torchvision
import time


class TactilePoseNet(nn.Module):
    def __init__(self):
        super(TactilePoseNet, self).__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        
        self.pool = nn.AvgPool2d(kernel_size=(9, 12))
        self.mlp = nn.Linear(512, 6)  # nn.Linear(512, 6)
    
    def forward(self, x):  # B, 3, 256, 256
        e1 = self.layer1(x)  # B, 64, 128, 128
        e2 = self.layer2(e1)  # B, 64, 64, 64
        e3 = self.layer3(e2)  # B, 128, 32, 32
        e4 = self.layer4(e3)  # B, 256, 16, 16
        e5 = self.layer5(e4)  # B, 512, 8, 8
        f = self.pool(e5).reshape(-1, 512)  # B, 512

        output = self.mlp(f)  # B, 6
        return output


if __name__ == "__main__":
    net = TactilePoseNet().to("cuda:0")
    print(net.parameters)
    data = torch.randn((32, 3, 288, 384)).to("cuda:0")
    output = net(data)
