import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import sys

import utils

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--exp', type=str, required=True, metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--num', type=int, default=None, metavar='N',
                    help='version number of the model (default: None)')
parser.add_argument('--arch', type=str, default=None, metavar='M',
                    help='model architecture (default: None)')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
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
from data import train_data_transforms, val_data_transforms

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
optimizer = optim.Adam(model.parameters(), lr=args.lr)
if 'optim_dict' in state.keys():
    optimizer.load_state_dict(state['optim_dict'])

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    logger.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset), accuracy))

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
