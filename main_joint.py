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
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.bs, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.bs, shuffle=False, num_workers=2)
    
    if config.net == 'vgg16':
        model_cls = vgg16_bn(num_classes=num_classes)
    elif config.net == 'resnet18':
        model_cls = resnet18(num_classes=num_classes)
    
    model_cls = model_cls.to(config.device)


    model_sr = _NetG()
    # model_sr.load_state_dict(torch.load(f'./model_sr_{config.dataset}/srresnet.pth', map_location=torch.device('cpu')))
    model_sr = model_sr.to(config.device)

    criterion = nn.CrossEntropyLoss()

    sr_criterion = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = torch.optim.SGD(list(model_sr.parameters()) + list(model_cls.parameters()), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    best_accuracy = 0.0

    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_test_losses = []
    epoch_test_accuracy = []
    epoch_test_psnr = []
    for epoch in range(config.num_epoch):
        model_cls.train()
        model_sr.train()
        epoch_loss = 0.0
        train_accuracy = 0.0
        for data, data_hr, lbl in tqdm(trainloader):
            optimizer.zero_grad()
            data, data_hr, lbl = data.to(config.device), data_hr.to(config.device), lbl.to(config.device)
            sr_output = model_sr(data)
            sr_loss = sr_criterion(sr_output, data_hr)
            cls_output = model_cls(sr_output)
            cls_loss = criterion(cls_output, lbl)
            loss = cls_loss + config.sr_weight * sr_loss
            loss.backward()
            epoch_loss += loss.item() * len(lbl) / len(trainset)
            _, train_pred = cls_output.detach().max(axis=-1)
            train_accuracy += (train_pred == lbl.detach()).sum().item() / len(trainset)
            optimizer.step()
        
        train_accuracy = train_accuracy * 100
        epoch_train_losses.append([epoch, epoch_loss])
        epoch_train_accuracy.append([epoch, train_accuracy])

        model_cls.eval()
        model_sr.eval()
        accuracy = 0.0
        epoch_test_loss = 0.0
        psnr = []
        with torch.no_grad():
            for test_data, test_data_hr, test_lbl in testloader:
                test_sr_output = model_sr(test_data.to(config.device))
                test_sr_loss = sr_criterion(test_sr_output, test_data_hr.to(config.device))
                test_cls_output = model_cls(test_sr_output)
                test_cls_loss = criterion(test_cls_output, test_lbl.to(config.device))
                test_loss = test_cls_loss + config.sr_weight * test_sr_loss
                epoch_test_loss += test_loss.item() * len(test_lbl) / len(testset)
                _, pred = test_cls_output.max(axis=-1)
                accuracy += (pred == test_lbl.to(config.device)).sum().item() / len(testset)
                psnr.append(compute_psnr(test_sr_output.cpu().detach(), test_data_hr.detach(), inv_norm))
        accuracy = accuracy * 100
        # print(len(testset))
        print(f'Epoch: {epoch}, train loss: {epoch_loss}, train accuracy: {train_accuracy}, test loss: {epoch_test_loss}, test accuracy: {accuracy}, psrn: {np.array(psnr).mean()}.')
        epoch_test_accuracy.append([epoch, accuracy])
        epoch_test_losses.append([epoch, epoch_test_loss])
        epoch_test_psnr.append([epoch, np.array(psnr).mean()])

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            saved_psnr = np.array(psnr).mean()
            best_epoch = epoch
            
            if not os.path.exists(f'./model_srCls_{config.dataset}_jointTraining'):
                os.mkdir(f'./model_srCls_{config.dataset}_jointTraining')
            
            torch.save(model_sr.state_dict(), f'./model_srCls_{config.dataset}_jointTraining/srresnet.pth')
            torch.save(model_cls.state_dict(), f'./model_srCls_{config.dataset}_jointTraining/{config.net}.pth')
            print(f'Model saved, accuracy: {best_accuracy}, psnr: {saved_psnr}.')
        

        scheduler.step()
    
    print(best_accuracy)
    print(saved_psnr)
    print(best_epoch)


    torch.save(model_sr.state_dict(), f'./model_srCls_{config.dataset}_jointTraining/srresnet_{epoch+1}_{np.array(psnr).mean()}.pth')
    torch.save(model_cls.state_dict(), f'./model_srCls_{config.dataset}_jointTraining/{config.net}_{epoch+1}_{accuracy}.pth')
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.array(epoch_train_losses)[:, 0], np.array(epoch_train_losses)[:, 1], color='r')
    plt.plot(np.array(epoch_test_losses)[:, 0], np.array(epoch_test_losses)[:, 1], color='b')
    plt.title('loss')
    plt.subplot(1, 3, 2)
    plt.plot(np.array(epoch_train_accuracy)[:, 0], np.array(epoch_train_accuracy)[:, 1], color='r')
    plt.plot(np.array(epoch_test_accuracy)[:, 0], np.array(epoch_test_accuracy)[:, 1], color='b')
    plt.title('accuracy')
    plt.subplot(1, 3, 3)
    plt.plot(np.array(epoch_test_psnr)[:, 0], np.array(epoch_test_psnr)[:, 1], color='b')
    plt.title('psnr')
    plt.savefig(f'./model_srCls_{config.dataset}_jointTraining/loss_acc_psnr.png')

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
    parser.add_argument('--net', default='vgg16', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--seed', default=1, type=int)


    parser.add_argument('--sr_weight', default=0.1, type=float)

    
    args = parser.parse_args()
    print(args)

    # set_seed(args.seed)


    train(config=args)