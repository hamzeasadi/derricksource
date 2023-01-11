import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary
import cv2

from torch import optim


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelBase(nn.Module):
    def __init__(self, name, created_time):
        super(ModelBase, self).__init__()
        self.name = name
        self.created_time = created_time

    def copy_params(self, state_dict):
        own_state = self.state_dict()
        for (name, param) in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

    def boost_params(self, scale=1.0):
        if scale == 1.0:
            return self.state_dict()
        for (name, param) in self.state_dict().items():
            self.state_dict()[name].copy_((scale * param).clone())
        return self.state_dict()

    # self - x
    def sub_params(self, x):
        own_state = self.state_dict()
        for (name, param) in x.items():
            if name in own_state:
                own_state[name].copy_(own_state[name] - param)

    # self + x
    def add_params(self, x):
        a = self.state_dict()
        for (name, param) in x.items():
            if name in a:
                a[name].copy_(a[name] + param)


class Constlayer(ModelBase):
    """
    doc
    """
    def __init__(self, name='constlayer', created_time=None, scale=1.0, outch=1, ks=5):
        super().__init__(name=name, created_time=created_time)
        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[outch, 1, ks, ks]), requires_grad=True)
        self.scale = scale
        self.outch = outch
        self.ks = ks
        
    def normalize(self):
        cntrpxl = int(self.ks/2)
        centeral_pixels = (self.const_weight[:, 0, cntrpxl, cntrpxl])
        for i in range(self.outch):
            sumed = (self.const_weight.data[i].sum() - centeral_pixels[i])/self.scale
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, cntrpxl, cntrpxl] = -self.scale

    def forward(self, x):
        self.normalize()
        x = F.conv2d(x, self.const_weight)
        return x.squeeze(), self.const_weight




class Constlayer1(nn.Module):
    """
    doc
    """
    def __init__(self, scale=1.0, outch=1, ks=5):
        super().__init__()
        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[outch, 1, ks, ks]), requires_grad=True)
        self.ks = ks
        self.scale = scale
        self.outch = outch

    def normalize(self):
        cntrpxl = int(self.ks/2)
        centeral_pixels = (self.const_weight[:, 0, cntrpxl, cntrpxl])
        for i in range(self.outch):
            sumed = (self.const_weight.data[i].sum() - centeral_pixels[i])/self.scale
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, cntrpxl, cntrpxl] = -self.scale

    def forward(self, x):
        # self.normalize()
        x1 = F.conv2d(x[:, 0:1, :, :], self.const_weight)
        x2 = F.conv2d(x[:, 1:2, :, :], self.const_weight)
        return x1.squeeze(), torch.exp(x2+5).squeeze(), self.const_weight



def create_data(h, w, c, b):
    x = torch.ones((b, c, h, w))
    ch = int(h/2)
    cw = int(w/2)
    for i in range(b):
        x[i, 1] = torch.zeros((h, w))
        x[i, 1, ch, cw] = 1.0
    return x


def train(net: nn.Module, epochs=10):
    opt = optim.Adam(params=net.parameters(), lr=3e-3)
    criterion = nn.SmoothL1Loss()
    data = create_data(h=3, w=3, b=10, c=2)
    y = torch.zeros(10)
    for epoch in range(epochs):
        out1, out2, w = net(data)
        # print(out1.shape, out2.shape)
        loss = criterion(out1, y) + criterion(out2, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch%50 == 0:
            print(loss.item(), w.squeeze().sum())
            print(w.squeeze())


def main():
    model = Constlayer1(scale=1000, outch=1, ks=3)
    train(net=model, epochs=1000)
    # data = create_data(b=2, c=2, h=3, w=3)
    # out1, out2, w = model(data)
    # print(out1, out2)
    # print(w.squeeze())

if __name__ == '__main__':
    main()