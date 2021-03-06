import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchprof
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os
import time

from models import *
from vgg_ghost import *
from vgg_ghost_2 import *
from vgg_ghost_v2 import *
from vgg_ghost_v2_2 import *
from vgg_ghost_v2_3 import *
from vgg_ghost_v3 import *
from vgg_ghost_v4 import *
from torchsummary import summary
from thop import profile
from thop import clever_format

#device = 'cpu' if torch.cuda.is_available() else 'cuda'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
Accuracy_list=[]

# init
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
criterion = nn.CrossEntropyLoss()

# Data
def Data_init():
    print('==> Preparing data..')
    #for train
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,testloader

def Model_init():
    # Model
    print('==> Building model..')
    #net = VGG('VGG16')
    net = VGG_ghost('VGG16')
    #net = VGG_ghost_2('VGG16')
    #net = VGG_ghost_v2('VGG16_Ghost_bottle')
    #net = VGG_ghost_v2_2('VGG16_Ghost_bottle')
    #net = VGG_ghost_v2_3('VGG16_Ghost_bottle')
    #net = VGG_ghost_v3('VGG16_Ghost_bottle')
    #net = VGG_ghost_v4('VGG16_Ghost_bottle')

    '''
    Flops
    '''
    '''
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops is {}'.format(flops))
    print('params is {}'.format(params))
    '''

    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    ''' Here to see torchprof'''
    x = torch.rand([1, 3, 32, 32]).cuda()
    # `profile_memory` was added in PyTorch 1.6, this will output a runtime warning if unsupported.
    with torchprof.Profile(net, use_cuda=True, profile_memory=True) as prof:
        net(x)
    # equivalent to `print(prof)` and `print(prof.display())`
    #print(prof.display(show_events=False))
    print(prof.display(show_events=True))
    '''Done'''

    criterion = nn.CrossEntropyLoss()
    return net

# Training
def train(epoch,net):
    print('Epoch {}/{}'.format(epoch + 1, 200))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))

def test(epoch,net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        Accuracy_list.append(100.*correct/total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/ckpt.pth')
        best_acc = acc

#------------------------------------------------------------------
# Loading weight files to the model and testing them.
def load_evaluate():
    net_test = VGG('VGG16')
    net_test = net_test.to(device)
    net_test = torch.nn.DataParallel(net_test)

    net_test.load_state_dict(torch.load('./checkpoint/ckpt.pth'))

    net_test.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_test(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

if __name__ == '__main__':
    trainloader,testloader = Data_init()
    net = Model_init()
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    #lr = scheduler.get_last_lr()
    #summary(net, (3, 32, 32))
    for epoch in range(50):
        train(epoch,net)
        test(epoch,net)
        print(best_acc)
        lr = scheduler.get_last_lr()
        scheduler.step()
        #print(optimizer)
    x1 = range(0, 50)
    y1 = Accuracy_list
    plt.plot(x1, y1, 'y-')
    plt.title('Accuracy trend')
    plt.ylabel('Accuracy')
    plt.show()
    print(y1)
