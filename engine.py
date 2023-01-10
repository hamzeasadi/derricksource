import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
import conf as cfg
import os



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model: nn.Module, data: DataLoader, criterion: nn.Module, optimizer: optim):
    epoch_error = 0
    l = len(data)
    model.train()
    for i, (X, Y) in enumerate(data):
        X = X.to(dev)
        Y = Y.to(dev)
        out, noise = model(X)
        loss = criterion(out, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_error += loss.item()
        # break
    return epoch_error/l


def val_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X = X.to(dev)
            Y = Y.to(dev)
            out, noise = model(X)
            loss = criterion(out, Y)
            epoch_error += loss.item()
            # break
    return epoch_error/l


def test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    model.to(dev)
    Y_true = torch.tensor([1], device=dev)
    Y_pred = torch.tensor([1], device=dev)
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X = X.to(dev)
            Y = Y.to(dev)
            out, noise   = model(X)
            yhat = torch.argmax(out, dim=1)
            Y_true = torch.cat((Y_true, Y))
            Y_pred = torch.cat((Y_pred, yhat))

    print(Y_pred.shape, Y_true.shape)

    acc = accuracy_score(Y_pred.cpu().detach().numpy(), Y_true.cpu().detach().numpy())
    print(f"acc is {acc}")


def local_test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    model.to(dev)
    Y_true = torch.tensor([1], device=dev)
    Y_pred = torch.tensor([1], device=dev)
    Noise, Real = 0, 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X = X.to(dev)
            Y = Y.to(dev)
            Real = X
            out, noise   = model(X)
            Noise = noise
            yhat = torch.argmax(out, dim=1)
            Y_true = torch.cat((Y_true, Y))
            Y_pred = torch.cat((Y_pred, yhat))
            # break


    ytrue = Y_true.cpu().detach().numpy()
    ypred = Y_pred.cpu().detach().numpy()
    print(Y_pred.shape, Y_true.shape)
    acc = accuracy_score(ytrue, ypred)
    print(f"acc is {acc}")

    cnfmtx = confusion_matrix(ytrue, ypred)
    plt.imshow(cnfmtx)
    plt.savefig(os.path.join(cfg.paths['model'], 'confusionl.png'))

    noise = Noise[0:2].detach().cpu()
    real = Real[0:2].squeeze().detach().cpu()
    print(real.shape, noise.shape)
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    for i in range(2):
        axs[i, 0].imshow(real[i], cmap='gray')
        axs[i, 0].axis('off')
        for j in range(1, 4):
            axs[i,j].imshow(noise[i, j-1], cmap='gray')
            axs[i, j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    
    filepath = os.path.join(cfg.paths['model'], f'liebherr.png')
    plt.savefig(filepath, bbox_inches='tight')


def main():
    y = np.random.randint(low=0, high=3, size=(20,))


if __name__ == '__main__':
    main()