import argparse
import os

import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageNet
from srresnet import _NetG
from models import *
from utils import *


def train(config):
    if config.dataset == 'tinyImageNet':
        transform_train = Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        inv_norm = transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5))

        training_subset = ImageNet(root='./OST300/train', train=True, transform=transform_train)
        testing_subset = ImageNet(root='./OST300/val', train=False, transform=transform_test)
        num_classes = 7
    else:
        raise NotImplementedError

    train_loader = DataLoader(training_subset, batch_size=config.bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(testing_subset, batch_size=config.bs, shuffle=False, num_workers=2)

    if config.net == 'vgg16':
        model_cls = vgg16_bn(num_classes=num_classes)
    elif config.net == 'resnet18':
        model_cls = resnet18(num_classes=num_classes)
    else:
        raise NotImplementedError

    model_cls = model_cls.to(config.device)

    model_sr = _NetG()
    model_sr.load_state_dict(torch.load('./model_sr/SRResNet_final.pth', map_location=torch.device('cpu')))
    model_sr = model_sr.to(config.device)

    cls_criterion = nn.CrossEntropyLoss()
    sr_criterion = nn.MSELoss()

    param_dicts = [
        {'params': list(model_cls.parameters())},
        {'params': list(model_sr.parameters()),
         'lr': config.lr * 0.1}
    ]

    optimizer = torch.optim.SGD(param_dicts, lr=config.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    best_test_accuracy = 0.0

    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_test_losses = []
    epoch_test_accuracy = []
    epoch_test_psnr = []
    for epoch in range(config.num_epoch):
        model_cls.train()
        model_sr.train()
        train_accuracy = 0.0
        epoch_train_loss = 0.0
        for data_lr, data_hr, label in tqdm(train_loader):
            optimizer.zero_grad()
            data_lr, data_hr, label = data_lr.to(config.device), data_hr.to(config.device), label.to(config.device)
            sr_output = model_sr(data_lr)
            sr_loss = sr_criterion(sr_output, data_hr)
            cls_output = model_cls(sr_output)
            cls_loss = cls_criterion(cls_output, label)
            loss = cls_loss + config.sr_weight * sr_loss
            loss.backward()
            epoch_train_loss += loss.item() * len(label) / len(training_subset)
            _, train_pred = cls_output.detach().max(axis=-1)
            train_accuracy += (train_pred == label.detach()).sum().item() / len(training_subset)
            optimizer.step()

        train_accuracy = train_accuracy * 100
        epoch_train_losses.append([epoch, epoch_train_loss])
        epoch_train_accuracy.append([epoch, train_accuracy])

        model_cls.eval()
        model_sr.eval()
        test_accuracy = 0.0
        epoch_test_loss = 0.0
        psnr = []
        with torch.no_grad():
            for test_data_lr, test_data_hr, test_label in test_loader:
                test_sr_output = model_sr(test_data_lr.to(config.device))
                test_sr_loss = sr_criterion(test_sr_output, test_data_hr.to(config.device))
                test_cls_output = model_cls(test_sr_output)
                test_cls_loss = cls_criterion(test_cls_output, test_label.to(config.device))
                test_loss = test_cls_loss + config.sr_weight * test_sr_loss
                epoch_test_loss += test_loss.item() * len(test_label) / len(testing_subset)
                _, test_pred = test_cls_output.max(axis=-1)
                test_accuracy += (test_pred == test_label.to(config.device)).sum().item() / len(testing_subset)
                psnr.append(compute_psnr(test_sr_output.cpu().detach(), test_data_hr.detach(), inv_norm))

        test_accuracy = test_accuracy * 100
        epoch_test_losses.append([epoch, epoch_test_loss])
        epoch_test_accuracy.append([epoch, test_accuracy])
        epoch_test_psnr.append([epoch, np.array(psnr).mean()])

        print(
            'Epoch: {}, trn_ls: {:.3f}, trn_acc: {:.1f}%, tst_ls: {:.3f}, tst_acc: {:.1f}%, tst_psnr: {:.2f}'.
            format(epoch + 1, epoch_train_loss, train_accuracy, epoch_test_loss, test_accuracy, np.array(psnr).mean()))

        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy

            if not os.path.exists('./model_joint'):
                os.mkdir('./model_joint')

            torch.save(model_sr.state_dict(), './model_joint/SRResNet_best.pth')
            torch.save(model_cls.state_dict(), './model_joint/VGG_best.pth')
            print('Model saved')

        scheduler.step()

    torch.save(model_sr.state_dict(), './model_joint/SRResNet_final.pth')
    torch.save(model_cls.state_dict(), './model_joint/VGG_final.pth')

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
    plt.savefig('./model_joint/loss_acc_psnr.png')


def compute_psnr(sr, hr, inv_norm):
    sr = inv_norm(sr)
    sr = torch.clamp(sr, min=0.0, max=1.0)
    sr = sr * 255.0
    hr = inv_norm(hr)
    hr = torch.clamp(hr, min=0.0, max=1.0)
    hr = hr * 255.0

    mse = torch.pow(torch.flatten(sr.double(), 1) - torch.flatten(hr.double(), 1), 2).mean(dim=1)
    psnr = 10.0 * torch.log10(255.0 ** 2 / (mse + 1e-10))

    return psnr.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tinyImageNet', type=str)
    parser.add_argument('--net', default='vgg16', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--sr_weight', default=0.5, type=float)

    args = parser.parse_args()

    train(config=args)
