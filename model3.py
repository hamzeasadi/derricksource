import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary
import cv2

from torch import optim


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, groups=1, bias=False, scale=10000):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.scale = scale

    def forward(self, x):
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = 0.0
        summed = torch.sum(self.weight.data, dim=(2,3), keepdim=True)/self.scale
        self.weight.data = self.weight.data/summed
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = -self.scale
        return super(CustomConv2d, self).forward(x)


class ConstConv(nn.Module):
    """
    doc
    """
    def __init__(self,  num_cls=10):
        super().__init__()
        self.num_cls = num_cls
        self.constlayer = CustomConv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.coord = self.coords(h=480, w=800)
        self.fx = self.feat()

    def coords(self, h, w):
        channelx = torch.ones(size=(h, w), device=dev)
        for i in range(h):
            channelx[i, :] = i*channelx[i, :]
        channelx = 2*(channelx/h) - 1

        channely = torch.ones(size=(h, w), device=dev)
        for i in range(w):
            channely[:, i] = i*channely[:, i]
        channely = 2*(channely/w) - 1

        return torch.cat((channelx.unsqueeze_(dim=0), channely.unsqueeze_(dim=0)), dim=0)

    def add_pos(self, res):
        Z = []
        for i in range(res.shape[0]):
            residual = res[i, :, :, :]
            z = torch.cat((residual, self.coord), dim=0)
            Z.append(z.unsqueeze_(dim=0)) 
        return torch.cat(tensors=Z, dim=0)

    def feat(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2), nn.BatchNorm2d(num_features=96), 
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, stride=1, groups=96), nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, groups=64), nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, groups=128), nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(num_features=256), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=256), nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1), 
            nn.BatchNorm2d(num_features=512), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, groups=512), nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(num_features=1024), nn.Tanh(), nn.AvgPool2d(kernel_size=(2, 7), stride=1),

            nn.Flatten(),

            nn.Linear(in_features=1024, out_features=self.num_cls)

        )

        return layer

    def forward(self, x):
        cordx = self.add_pos(res=x)
        nx = self.constlayer(cordx)
        out = self.fx(nx)
        return out, nx


def main():
    x = torch.randn(size=(5, 1, 480, 800))
    model = ConstConv(num_cls=6)
    out, noise = model(x)
    print(out.shape, out)
    print(noise.shape)
    # data = create_data(b=2, c=2, h=3, w=3)
    # out1, out2, w = model(data)
    # print(out1, out2)
    # print(w.squeeze())

if __name__ == '__main__':
    main()