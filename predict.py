#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 07:33:51 2019

@author: hnagaty
"""
#%%
top_k = 3
category_names = "cat_to_name.json"

#%%
import torch
import numpy as np
from PIL import Image
import json
from torchvision import models
import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument("image", help="A flower image to predict")
parser.add_argument("checkpoint", help="A saved model to use")
parser.add_argument("--top_k", help="No. of classes", type = int)
parser.add_argument("--category_names", help="File with category names")
parser.add_argument("--gpu", help="Use GPU", action="store_true")

args = parser.parse_args()

imagefile = args.image
checkpoint = args.checkpoint
if args.top_k is not None: top_k = args.top_k
category_names = args.category_names

gpu = args.gpu
device = torch.device("cuda" if gpu else "cpu")
if (gpu):
    print("Using GPU ........")
else:
    print("Using CPU ........")

#%%
def load_checkpoint(filepath):
    if gpu:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'densenet':
        model = models.densenet121(pretrained=True)
    else:
        print(f"{checkpoint['arch']} architecture is not supported. ")
        exit()
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    h = im.height
    w = im.width
    f = w/h
    if f<= 1:
        im0 = im.resize((256, int(256/f)))
    else:
        im0 = im.resize((int(256*f), 256))
    im1 = im0.crop(box=(16, 16, 240, 240))

    npImg1 = np.array(im1)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    npImg1 = npImg1/255
    npImg2 = (npImg1-mean)/std
    npImg3 = npImg2.transpose((2,0,1))

    return(npImg3)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    im = Image.open(image_path)
    npImg = process_image(im)
    
    img = torch.from_numpy(npImg)
    img = img.view(1, 3, 224, 224)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)
        
    top_p, top_idx = ps.topk(topk, dim=1)
    top_p = top_p.tolist()[0]
    top_idx = top_idx.tolist()[0]
    
    inverseMap = {val: key for key, val in
            model.class_to_idx.items()
            }
    top_class = [inverseMap[item] for item in top_idx]
    
    return (top_p, top_class)
#%%


#%%
model = load_checkpoint(checkpoint)

#%%
probs, classes = predict(imagefile, model, topk=top_k)
if category_names is not None:
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)    
    flowers = [cat_to_name[x] for x in classes]
else:
    flowers = classes
print(f"Highest {top_k} flower types ranked in descending order:")

for f,p in zip(flowers,probs):
    print(f'{f} \t with probability {p:.2}'.format(f,p))