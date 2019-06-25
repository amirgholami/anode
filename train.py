#*
# @file ANODE training driver based on arxiv:1902.10298
# This file is part of ANODE library.
#
# ANODE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ANODE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ANODE.  If not, see <http://www.gnu.org/licenses/>.
#*
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
import math
import sys
import os


parser = argparse.ArgumentParser()
parser.add_argument('--network', type = str, choices = ['resnet', 'sqnxt'], default = 'sqnxt')
parser.add_argument('--method', type = str, choices=['Euler', 'RK2', 'RK4'], default = 'Euler')
parser.add_argument('--num_epochs', type = int, default = 350)
parser.add_argument('--lr', type=float, default = 0.1)
parser.add_argument('--Nt', type=int, default = 2)
parser.add_argument('--batch_size', type = int, default = 256)
args = parser.parse_args()
if args.network == 'sqnxt':
    from models.sqnxt import SqNxt_23_1x, lr_schedule
    writer = SummaryWriter('sqnxt/' + args.method + '_lr_' + str(args.lr) + '_Nt_' + str(args.Nt) + '/')
elif args.network == 'resnet':
    from models.resnet import ResNet18, lr_schedule
    writer = SummaryWriter('resnet/' + args.method + '_lr_' + str(args.lr) + '_Nt_' + str(args.Nt) + '/')

from anode import odesolver_adjoint as odesolver

num_epochs = int(args.num_epochs)
lr           = float(args.lr)
start_epoch  = 1
batch_size   = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")
best_acc    = 0.

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'Nt':int(args.Nt)})
        self.options.update({'method':args.method})
        print(self.options)

    def forward(self, x):
        out = odesolver(self.odefunc, x, self.options)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        

# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_train, train = True, download = True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_test, train = False, download = True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 4, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, num_workers = 4, shuffle = False)

if args.network == 'sqnxt':
    net = SqNxt_23_1x(10, ODEBlock)
elif args.network == 'resnet':
    net = ResNet18(ODEBlock)

net.apply(conv_init)
print(net)
if is_use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


def train(epoch):
    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    optimizer  = optim.SGD(net.parameters(), lr = lr_schedule(lr, epoch), momentum = 0.9, weight_decay = 5e-4)
    
    print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Loss', loss.item(), epoch* 50000 + batch_size * (idx + 1)  )
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(train_dataset) // batch_size, 
                          train_loss / (batch_size * (idx + 1)), correct / total))
        sys.stdout.flush()
    writer.add_scalar('Train/Accuracy', correct / total, epoch )
        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        test_loss  += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        writer.add_scalar('Test/Loss', loss.item(), epoch* 50000 + test_loader.batch_size * (idx + 1)  )
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size, 
                          test_loss / (100 * (idx + 1)), correct / total))
        sys.stdout.flush()
    writer.add_scalar('Test/Accuracy', correct / total, epoch )
        
for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()
    train(_epoch)
    print()
    test(_epoch)
    print()
    print()
    end_time   = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))
print('Best Acc@1: %.4f' % (best_acc * 100))
writer.close()
