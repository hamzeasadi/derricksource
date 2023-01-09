import os
import conf as cfg
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


batch_size = 128

liebherrtrain = ImageFolder(root=cfg.paths['liebherrtrain'])
liebherrtest = ImageFolder(root=cfg.paths['liebherrtest'])

visiontrain = ImageFolder(root=cfg.paths['visiontrain'])
visiontest = ImageFolder(root=cfg.paths['visiontest'])

def create_loader(dataset, batch_size, train_percent=0.85):
    l = len(dataset)
    train_size = int(train_percent*l)
    valid_size = l - train_size
    train_data, valid_data = random_split(dataset=dataset, lengths=[train_size, valid_size])
    return DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(valid_data, batch_size=batch_size, shuffle=True)


# liebherr
ltrain_loader, lvalid_loader = create_loader(dataset=liebherrtrain, batch_size=batch_size)
ltest_loader = DataLoader(liebherrtest, batch_size=batch_size)

# vision
vtrainl, vvalidl = create_loader(dataset=visiontrain, batch_size=batch_size)
vtestl = DataLoader(dataset=visiontest, batch_size=batch_size)

dataloader = dict(liebherr=(ltrain_loader, lvalid_loader, ltest_loader), vision=(vtrainl, vvalidl, vtestl))

def main():
    pass


if __name__ == "__main__":
    main()
