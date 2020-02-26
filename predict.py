#Author: Udit Maherwal

import argparse
import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F
import torchvision
import json 
import os
from torchvision import datasets,transforms,models
from collections import OrderedDict
import time
import PIL
from PIL import Image
import seaborn as sb
import train

def arguments_parser():
    parser = argparse.ArgumentParser("Flowers Prediction - Probabilities and image class")
    parser.add_argument('--checkpoint_01',dest='checkpoint', help="stored the snapshot of your trained model", default="checkpoint_01.pth")
    parser.add_argument('--image_path',help="image for which we have to find probabilities and image_class",default ="flowers/test/1/image_06752.jpg" )
    parser.add_argument('--device',dest='gpu',type=str,help ="Either GPU(Faster) or  CPU(Prefer only if you have GPU driver installed)",default='gpu')
    parser.add_argument('--k',dest='k',help='top labels taken till',default=5)
    return parser.parse_args()

def process_image(image):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im=Image.open(image)
    
    width, height = im.size
    
    if height > width :
        scaling = 256 , 256 * (height/width)
    else:
        scaling = 256 * (width/height) , 256
        
    im.thumbnail(scaling,Image.ANTIALIAS)
    
    left = (width/4 - 112)
    top = (height/4 - 112) 
    right = (width/4 + 112)
    bottom = (height/4 + 112)

    im = im.crop((left, top, right, bottom))
     
    color_channel = (np.array(im))/255
    
    means = [0.485, 0.456, 0.406]
    deviation = [0.229, 0.224, 0.225]
    nomralized_image = ((color_channel - means)/deviation).transpose((2,0,1))
    
    return nomralized_image

def predict(image_path,model,k=5):
    '''Predict the class (or classes) of an image using a trained deep learning model.
    ''' 
    model.cuda()
    image = torch.from_numpy(np.expand_dims(process_image(image_path),axis=0)).type(torch.FloatTensor).to('cuda')
    output = model(image)
    probs, labels = output.topk(k)
    probs= torch.exp(probs)
    
    probs = np.array(probs.detach())[0]
    labels = np.array(labels.detach())[0]
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    tlabels = [idx_to_class[label] for label in labels]
    
    return probs,tlabels

def main():
    arguments = arguments_parser()
    i=0
    gpu = arguments.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomRotation(30),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([.485,.456,.406],
                                                                   [.229,.224,.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([.485,.456,.406],
                                                                     [.229,.224,.225])])

    testing_transfroms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([.485,.456,.406],
                                                                  [.229,.224,.225])])

    #Loading the datasets with ImageFolder
    image_datasets = [datasets.ImageFolder(train_dir,transform=training_transforms),
                      datasets.ImageFolder(valid_dir,transform=validation_transforms),
                      datasets.ImageFolder(test_dir,transform=testing_transfroms)]

    #Using the image datasets and the trainforms, defining the dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0],batch_size=64,shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1],batch_size=64,shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2],batch_size=64,shuffle=True)]
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    load_from = arguments.checkpoint
    model= train.load_checkpoint(load_from)
    image = arguments.image_path
    k = arguments.k
    model.class_to_idx = image_datasets[0].class_to_idx
    probabilities,image_class = predict(image,model,k)
    labels = [cat_to_name[str(index)] for index in image_class]
    probs = probabilities
    print('Image: ' + image)
    i=0 
    while i < 5:
        print("{} with a probability of {}".format(labels[i], probs[i]))
        i += 1 
    
if __name__ == "__main__":
    main()

