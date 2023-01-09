import os, sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import time
from torch import nn as nn
from torch import optim
import conf as cfg



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KeepTrack():
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, minerror=0.1)

    def save_ckp(self, model: nn.Module, opt: optim.Optimizer, epoch, minerror, fname: str):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['minerror'] = minerror
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state


def savearray(data: list, fname:str):
    path = os.path.join(cfg.paths['model'], fname)
    with open(path, 'w') as f:
        for pair in data:
            f.write(pair+'\n')



def main():
    batch_size = 5

    
  

if __name__ == '__main__':
    main()


