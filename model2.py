import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary
import cv2


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
        x = F.conv2d(x, self.const_weight, padding='same')
        return x



class ConstConv(ModelBase):
    """
    doc
    """
    def __init__(self, scale=1, ks=5, outch=1, name='constlayer', created_time=None, num_cls=10):
        super().__init__(name=name, created_time=created_time)
        self.num_cls = num_cls
        self.cont1 = Constlayer(name='constlayer1', created_time=None, scale=10, outch=1, ks=5)
        self.cont2 = Constlayer(name='constlayer2', created_time=None, scale=100, outch=1, ks=5)
        self.cont3 = Constlayer(name='constlayer3', created_time=None, scale=1000, outch=1, ks=5)
        self.cont4 = Constlayer(name='constlayer4', created_time=None, scale=10000, outch=1, ks=5)
        self.coord = self.coords(h=480, w=800)
        self.fx = self.feat()

    def add_pos(self, res):
        Z = []
        for i in range(res.shape[0]):
            residual = res[i, :, :, :]
            z = torch.cat((residual, self.coord), dim=0)
            Z.append(z.unsqueeze_(dim=0)) 
        return torch.cat(tensors=Z, dim=0)

    
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


    def feat(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=4+2, out_channels=96, kernel_size=5, stride=2), nn.BatchNorm2d(num_features=96), 
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
        x1 = self.cont1(x)
        x2 = self.cont2(x)
        x3 = self.cont3(x)
        x4 = self.cont4(x)
        noise = torch.cat((x1, x2, x3, x4), dim=1)
        noisecoord = self.add_pos(res=noise)
        x = self.fx(noisecoord)
        
        return x, noise











def main():
    x = torch.randn(size=(3, 1, 480, 800))






if __name__ == '__main__':
    main()