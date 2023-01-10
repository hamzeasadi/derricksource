import numpy as np
from utils import KeepTrack, savearray  
import conf as cfg
import datasetup as ds
import model2
import engine
import argparse
import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
import os

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--data', '-d', type=str, required=True, default='None')
parser.add_argument('--modelname', '-mn', type=str, required=True, default='None')
parser.add_argument('--epoch', '-e', type=int, required=False, metavar='epoch', default=1)
parser.add_argument('--numcls', '-nc', type=int, required=True, metavar='numcls', default=10)

args = parser.parse_args()




def train(net, train_loader, val_loader, opt, criterion, epochs, minerror, modelname:str):

    kt = KeepTrack(path=cfg.paths['model'])
    losses = []
    for epoch in range(epochs):
        train_loss = engine.train_step(model=net, data=train_loader, criterion=criterion, optimizer=opt)
        val_loss = engine.val_step(model=net, data=val_loader, criterion=criterion)
        losses.append((train_loss, val_loss))

        if val_loss < minerror:
            minerror = val_loss
            kt.save_ckp(model=net, opt=opt, epoch=epoch, minerror=val_loss, fname=modelname)
            
        print(f"epoch={epoch}, train_loss={train_loss} val_loss={minerror}")

    savearray(data=losses, fname=f'{modelname}.txt')


def main():
    model_name = f"{args.modelname}.pt"
    keeptrack = KeepTrack(path=cfg.paths['model'])
    Net = model2.ConstConv(num_cls=args.numcls)
    Net.to(dev)
    opt = optim.Adam(params=Net.parameters(), lr=3e-4)
    criteria = nn.CrossEntropyLoss()
    train_loader, valid_loader, test_loader = ds.dataloader[args.data]
    # train_loader, valid_loader = ds.create_loader(dataset=dataset, batch_size=128, train_percent=0.85)
    minerror = np.inf
    # if False:
    if args.train:
        train(net=Net, train_loader=train_loader, val_loader=valid_loader, opt=opt, criterion=criteria, epochs=args.epoch, minerror=minerror, modelname=model_name)

    # if True:
    if args.test:
        # model_name = f"vision_residual_1.pt"
        state = keeptrack.load_ckp(fname=model_name)
        Net.load_state_dict(state['model'], strict=False)
        print(f"min error is {state['minerror']} which happen at epoch {state['epoch']}")
        # engine.test_step(model=Net, data=test_loader, criterion=criteria)
        engine.local_test_step(model=Net, data=test_loader, criterion=criteria)


if __name__ == '__main__':
    main()