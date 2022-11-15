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

        trainset = tinyImageNet(root='./data/tiny-imagenet-200/train', train=True, transform=transform_train, lr_cls=True)

        testset = tinyImageNet(root='./data/tiny-imagenet-200/val/val_imgs', train=False, transform=transform_test, lr_cls=True)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=config.bs, shuffle=False, num_workers=4)

        num_classes = 200
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.bs, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.bs, shuffle=False, num_workers=2)
    
    if config.net == 'vgg16':
        model = vgg16_bn(num_classes=num_classes)
    elif config.net == 'resnet18':
        model = resnet18(num_classes=num_classes)
    
    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    best_accuracy = 0.0

    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_test_losses = []
    epoch_test_accuracy = []
    for epoch in range(config.num_epoch):
        model.train()
        epoch_loss = 0.0
        train_accuracy = 0.0
        for data, data_hr, lbl in tqdm(trainloader):
            optimizer.zero_grad()
            data, lbl = data.to(config.device), lbl.to(config.device)
            output = model(data)
            loss = criterion(output, lbl)
            loss.backward()
            epoch_loss += loss.item() * len(lbl) / len(trainset)
            _, train_pred = output.detach().max(axis=-1)
            train_accuracy += (train_pred == lbl.detach()).sum().item() / len(trainset)
            optimizer.step()
        
        train_accuracy = train_accuracy * 100
        epoch_train_losses.append([epoch, epoch_loss])
        epoch_train_accuracy.append([epoch, train_accuracy])

        model.eval()
        accuracy = 0.0
        epoch_test_loss = 0.0
        with torch.no_grad():
            for test_data, test_data_hr, test_lbl in testloader:
                test_output = model(test_data.to(config.device))
                test_loss = criterion(test_output, test_lbl.to(config.device))
                epoch_test_loss += test_loss.item() * len(test_lbl) / len(testset)
                _, pred = test_output.max(axis=-1)
                accuracy += (pred == test_lbl.to(config.device)).sum().item() / len(testset)
        accuracy = accuracy * 100
        # print(len(testset))
        print(f'Epoch: {epoch}, train loss: {epoch_loss}, train accuracy: {train_accuracy}, test loss: {epoch_test_loss}, test accuracy: {accuracy}.')
        epoch_test_accuracy.append([epoch, accuracy])
        epoch_test_losses.append([epoch, epoch_test_loss])

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            
            if not os.path.exists(f'./model_lrCls_{config.dataset}'):
                os.mkdir(f'./model_lrCls_{config.dataset}')
            
            torch.save(model.state_dict(), f'./model_lrCls_{config.dataset}/{config.net}_best.pth')
            print(f'Model saved, accuracy: {best_accuracy}.')
        

        scheduler.step()
    
    print(best_accuracy)
    print(best_epoch)

    # Save model after the last epoch training. Please use this model for evaluation; don't use the saved best model.
    torch.save(model.state_dict(), f'./model_lrCls_{config.dataset}/{config.net}_{epoch+1}_{accuracy}.pth')
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.array(epoch_train_losses)[:, 0], np.array(epoch_train_losses)[:, 1], color='r')
    plt.plot(np.array(epoch_test_losses)[:, 0], np.array(epoch_test_losses)[:, 1], color='b')
    plt.title('loss')
    plt.subplot(1, 2, 2)
    plt.plot(np.array(epoch_train_accuracy)[:, 0], np.array(epoch_train_accuracy)[:, 1], color='r')
    plt.plot(np.array(epoch_test_accuracy)[:, 0], np.array(epoch_test_accuracy)[:, 1], color='b')
    plt.title('accuracy')
    plt.savefig(f'./model_lrCls_{config.dataset}/loss_acc.png')


if __name__ == '__main__':
    
    # print(torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tinyImageNet', type=str)
    parser.add_argument('--net', default='vgg16', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int)

    
    args = parser.parse_args()
    print(args)

    # set_seed(args.seed)


    train(config=args)


