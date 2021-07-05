 

import os
import numpy as np
import math

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from voc import VOCDataset

from resnet_yolo import resnet50
from torchvision import models
from torchsummary import summary

from loss import Loss
import pandas as pd


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#########################
# data
#########################

# Path to data dir.
train_image_dir = '/home/mithila/det_projs/data/VOC2012/JPEGImages'
test_image_dir  = '/home/mithila/det_projs/data/VOC2007/JPEGImages'

# Path to label files.
train_label = 'data/voc2012.txt'
val_label   = 'data/voc2007test.txt'


# Training hyper parameters.
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64

# Learning rate scheduling.
def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

##################################
# model backbone loading - YOLO
##################################
yolo = resnet50()
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
dd = yolo.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in dd.keys() and not k.startswith('fc'):
        print('yes')
        dd[k] = new_state_dict[k]
yolo.load_state_dict(dd)
yolo.cuda()
summary(yolo, input_size=(3, 448, 448))
 
##################################
# dataloader
##################################

# Setup loss and optimizer.
criterion = Loss()
optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

# Load Pascal-VOC dataset.
train_dataset = VOCDataset(True, train_image_dir, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = VOCDataset(False, test_image_dir, val_label)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Number of training images: ', len(train_dataset))

##################################
# where to save model
##################################

log_dir = 'results'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

##################################
# start training
##################################
results_dict = {'Epoch':[], 'train_loss':[], 'val_loss':[]}

best_val_loss = np.inf

for epoch in range(num_epochs):
    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))

    # Training.
    yolo.train()
    total_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(train_loader):
        # Update learning rate.
        update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
        lr = get_lr(optimizer)

        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        imgs, targets = imgs.cuda(), targets.cuda()

        # Forward to compute loss.
        preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        # Backward to update model weight.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current loss.
        print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
        % (epoch, num_epochs, i, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

        # TensorBoard.
        n_iter = epoch * len(train_loader) + i

    # Validation.
    yolo.eval()
    val_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(val_loader):
        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        imgs, targets = imgs.cuda(), targets.cuda()

        # Forward to compute validation loss.
        with torch.no_grad():
            preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss_this_iter = loss.item()
        val_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter
    val_loss /= float(total_batch)
    
    
    # update dict
    results_dict['Epoch'].append(epoch)
    results_dict['train_loss'].append(total_loss / float(total_batch))
    results_dict['val_loss'].append(val_loss)
    
    df = pd.DataFrame(results_dict)
    df.to_csv(os.path.join(log_dir,'history.csv'))
    
    # Save results.
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_best.pth'))

    # Print.
    print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
    % (epoch + 1, num_epochs, val_loss, best_val_loss))
