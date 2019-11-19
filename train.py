#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:51:30 2019

@author: hnagaty
"""

data_dir = '~/data/flower_data'
save_dir = '.'
arch = 'dense'
learning_rate = 0.01
hidden_units = 256
epochs = 1
gpu = True



#%% Imports here
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import time

#%%
# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="The folder that holds the images")
parser.add_argument("--save_dir", help="The save directory", default = '.')
parser.add_argument("--arch", help="The arcitecture to use. Default is DenseNet", default = 'densenet')
parser.add_argument("--learning_rate", help="Training learning rate. Default is 0.01", type = float, default = 0.1)
parser.add_argument("--hidden_units", help="No of hidden units. Default is 256", type = int, default = 256)
parser.add_argument("--epochs", help="Training epochs. Default is 20", type = int, default = 20)
parser.add_argument("--gpu", help="Use GPU if available. Default is yes", action="store_true")

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu




#%% Training
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#%%
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

#%%
# Import an pre-trained model
# Use GPU if it's available
if gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

model = models.densenet121(pretrained=True)

if arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif arch == 'densenet':
    model = models.densenet121(pretrained=True)
else:
    print(f"{arch} architecture is not supported. Please use 'densenet' or 'vgg13'")
    exit()

#%%
# define a new classifier (dense network)
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier1 = nn.Sequential(OrderedDict([ # I finally used this one
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.4)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))



classifier2 = nn.Sequential(OrderedDict([ # I finally used this one
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.4)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


if arch == 'vgg13':
    model.classifier = classifier1
else:
    model.classifier = classifier2

#%%
# defining loss & optimiser
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device);
train_losses, valid_losses  =  [], []
valid_acc = []

#%%
# training the model
steps = 0
running_loss = 0
print_every = 10

start = time.time()

print("============================================")
print("Now training the model with the below settings")
print(f"Model is trained on {device}")
print(f"Architecture: {arch}")
print(f"Learning Rate: {learning_rate}")
print(f"Hidden Layers: {hidden_units}")
print(f"Epochs: {epochs}")






for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}, Step {steps}... "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader)*100:.3f}%")
                            
            train_losses.append(running_loss/print_every)
            valid_losses.append(valid_loss/len(validloader))
            valid_acc.append(accuracy/len(validloader))
            running_loss = 0
            model.train()

end = time.time()
print(f"Total training time: {end-start:,.2f}s. That's {(end-start)/60:,.2f} minutes")

#%%
model.class_to_idx = train_data.class_to_idx

checkpoint = {'state_dict': model.state_dict(),
              'optim_state': optimizer.state_dict(),
              'epochs': epochs,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir + '/checkpointAs.pth')