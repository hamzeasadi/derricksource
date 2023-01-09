import os
import conf as cfg
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

batch_size = 128

trf = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[126/255], std=[200/255])])
liebherrtrain = ImageFolder(root=cfg.paths['liebherrtrain'], transform=trf)
liebherrtest = ImageFolder(root=cfg.paths['liebherrtest'], transform=trf)

visiontrain = ImageFolder(root=cfg.paths['visiontrain'], transform=trf)
visiontest = ImageFolder(root=cfg.paths['visiontest'], transform=trf)

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
