import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import sys
import torch.nn.functional as F

import utils

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--exp', type=str, required=True, metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--num', type=int, default=None, metavar='N',
                    help='version number of the model (default: None)')
parser.add_argument('--arch', type=str, default=None, metavar='M',
                    help='model architecture (default: None)')
parser.add_argument('--data', type=str, default='bird_dataset_ex', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wdecay', type=float, default=0.0001, metavar='M',
                    help='SGD weight decay (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status (default: 1)')
parser.add_argument('--no-train-data-aug', type=bool, default=False, action='store_true',
                    help='no data augmentation during training (default: False)')
parser.add_argument('--no-val-data-aug', type=bool, default=False, action='store_true',
                    help='no data augmentation during validation (default: False)')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Define experiment folder
exp_dir = 'experiments/' + args.exp

# Load state
try:
    state = utils.load_state(exp_dir)
except OSError:
    state = {
        'arch': args.arch,
        'start_epoch': 0,
        'best_acc': 0
    }

# Logger
logger = utils.get_logger(exp_dir)

# Data initialization and loading
from data import train_data_transforms, aug_train_data_transforms, val_data_transforms, aug_val_data_transforms

if not args.no_train_data_aug:
    train_data_transforms = aug_train_data_transforms
if not args.no_val_data_aug:
    val_data_transforms = aug_val_data_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=train_data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=val_data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Log model and command
logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))

# Model
model = utils.get_model(state)
if use_cuda:
    logger.info('Using GPU')
    model.cuda()
else:
    logger.info('Using CPU')
logger.info("{}\n".format(model))

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
if 'optim_dict' in state.keys():
    optimizer.load_state_dict(state['optim_dict'])

def train(epoch):
    model.train()

    full_correct = 0
    batch_correct = 0
    full_loss = 0
    batch_loss = 0
    full_len = 0
    batch_len = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        batch_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        batch_loss += loss.data.item()
        batch_len += len(data)

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                batch_loss/batch_len, batch_correct, batch_len, 100. * batch_correct/batch_len))
            full_correct += batch_correct
            batch_correct = 0
            full_loss += batch_loss
            batch_loss = 0
            full_len += batch_len
            batch_len = 0
    logger.info('Epoch Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
        full_loss/full_len, full_correct, full_len, 100. * full_correct/full_len))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for datas, target in val_loader:
        for i, data in enumerate(datas):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = F.softmax(model(data), dim=1)
            res = output if i == 0 else torch.max(res, output)
        pred = res.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    logger.info('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(val_loader.dataset), accuracy))

    return accuracy

start_epoch = state['start_epoch']
for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    accuracy = validation()

    is_best = accuracy > state['best_acc']

    state['start_epoch'] = epoch + 1
    state['best_acc'] = accuracy if is_best else state['best_acc']
    state['model_state'] = model.state_dict()
    state['optim_state'] = optimizer.state_dict()

    utils.save_state(state, is_best, exp_dir)
    logger.info('\nModel saved\n')
