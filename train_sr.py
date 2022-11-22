import torch
import torch.nn as nn
import torch.nn.functional as f

import torchvision
from torchvision import transforms

import numpy as np
import pickle as pkl
import os

import matplotlib.pyplot as plt
import random

from tqdm import tqdm
import argparse


from model import *
from srresnet import _NetG

from dataset import *

from utils import *


def train(config):
    
    if config.dataset == "tinyImageNet":
        transform_train = Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        inv_norm = transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1.0/0.5, 1.0/0.5, 1.0/0.5))

        trainset = tinyImageNet(root='./data/tiny-imagenet-200/train', train=True, transform=transform_train)

        testset = tinyImageNet(root='./data/tiny-imagenet-200/val/val_imgs', train=False, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=config.bs, shuffle=False, num_workers=4)

        num_classes = 200
    
    elif config.dataset == "CIFAR10":
        transform_train = Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        inv_norm = transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1.0/0.5, 1.0/0.5, 1.0/0.5))

        trainset = cifar10(root='./', train=True, download=True, transform=transform_train, lr_cls=False)

        testset = cifar10(root='./', train=False, download=True, transform=transform_test, lr_cls=False)

        num_classes = 10
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.bs, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.bs, shuffle=False, num_workers=2)
    
    model = _NetG()
    model = model.to(config.device)
    
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    best_psnr = 0.0

    epoch_train_losses = []
    epoch_test_losses = []
    epoch_test_psnr = []
    for epoch in range(config.num_epoch):
        model.train()
        epoch_loss = 0.0
        for data, data_hr, lbl in tqdm(trainloader):
            optimizer.zero_grad()
            data, data_hr = data.to(config.device), data_hr.to(config.device)
            output = model(data)
            loss = criterion(output, data_hr)
            loss.backward()
            epoch_loss += loss.item() * len(lbl) / len(trainset)
            optimizer.step()
        
        epoch_train_losses.append([epoch, epoch_loss])

        model.eval()
        epoch_test_loss = 0.0
        psnr = []
        with torch.no_grad():
            for test_data, test_data_hr, test_lbl in testloader:
                test_output = model(test_data.to(config.device))
                test_loss = criterion(test_output, test_data_hr.to(config.device))
                epoch_test_loss += test_loss.item() * len(test_lbl) / len(testset)
                psnr.append(compute_psnr(test_output.cpu().detach(), test_data_hr.detach(), inv_norm))
        # print(len(testset))
        print(f'Epoch: {epoch}, train loss: {epoch_loss}, test loss: {epoch_test_loss}, psnr: {np.array(psnr).mean()}.')
        epoch_test_psnr.append([epoch, np.array(psnr).mean()])
        epoch_test_losses.append([epoch, epoch_test_loss])

        if np.array(psnr).mean() >= best_psnr:
            best_psnr = np.array(psnr).mean()
            best_epoch = epoch
            
            if not os.path.exists(f'./model_sr_{config.dataset}'):
                os.mkdir(f'./model_sr_{config.dataset}')
            
            torch.save(model.state_dict(), f'./model_sr_{config.dataset}/{config.net}.pth')
            print(f'Model saved, psnr: {best_psnr}.')
        

        scheduler.step()
    print(best_psnr)
    print(best_epoch)
    

    # Save model after the last epoch training. Please use this saved model for evaluation; don't use the saved best model.
    torch.save(model.state_dict(), f'./model_sr_{config.dataset}/{config.net}_{epoch+1}_{np.array(psnr).mean()}.pth')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.array(epoch_train_losses)[:, 0], np.array(epoch_train_losses)[:, 1], color='r')
    plt.plot(np.array(epoch_test_losses)[:, 0], np.array(epoch_test_losses)[:, 1], color='b')
    plt.title('loss')
    plt.subplot(1, 2, 2)
    # plt.plot(np.array(epoch_train_accuracy)[:, 0], np.array(epoch_train_accuracy)[:, 1], color='r')
    plt.plot(np.array(epoch_test_psnr)[:, 0], np.array(epoch_test_psnr)[:, 1], color='b')
    plt.title('psnr')
    plt.savefig(f'./model_sr_{config.dataset}/MSEloss_psnr.png')

def compute_psnr(sr, hr, inv_norm):
    sr = inv_norm(sr)
    sr = torch.clamp(sr, min=0.0, max=1.0)
    sr = sr * 255.0
    hr = inv_norm(hr)
    hr = torch.clamp(hr, min=0.0, max=1.0)
    hr = hr * 255.0

    mse = torch.pow(torch.flatten(sr.double(), 1) - torch.flatten(hr.double(), 1), 2).mean(dim=1)
    psnr = 10.0 * torch.log10(255.0**2 / (mse+1e-10))

    return psnr.mean()

if __name__ == '__main__':
    
    # print(torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tinyImageNet', type=str)
    parser.add_argument('--net', default='srresnet', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--seed', default=1, type=int)

    
    args = parser.parse_args()
    print(args)

    # set_seed(args.seed)


    train(config=args)