import conf as cfg
import os
from matplotlib import pyplot as plt
import numpy as np








def plot_loss(path):
    with open(path, 'r') as f:
        filelines = f.readlines()
        print(filelines)






def main():
    path = os.path.join(cfg.paths['model'], 'derrickLiebherr.pt.txt')
    plot_loss(path)


if __name__ == '__main__':
    main()