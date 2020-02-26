#Author : Udit Maherwal

import argparse
import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms,models
from collections import OrderedDict
import time
import PIL
import seaborn as sb

def arguments_parser():
    parser = argparse.ArgumentParser("Flower Classifier Training")
    parser.add_argument('--arch', dest ='arch' ,default='vgg16',type = str,choices=["vgg16","densenet121"],help='Choose a model you want to work with')
    parser.add_argument('--learning_rate', dest='lr', type = float,default=0.001,help='Define a learning rate for this model')
    parser.add_argument('--epochs',type = int, dest='epo',default=3,help='Iterations you would like to give')
    parser.add_argument('--hidden_inputs',type = int, dest='hidden_inputs',default=4096,help='Hidden inputs for network')
    parser.add_argument('--device',dest='gpu',type=str,help ="Either GPU(Faster) or  CPU(Prefer only if you have GPU driver installed)",default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", help="stores the snapshot of your trained model", default="checkpoint_01.pth")
    parser.add_argument('--dropout',dest='dropout',default=0.5)
    return parser.parse_args()


    
def network(arch,architectures,learning_rate,device,dropout,hidden_inputs):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1',nn.Linear(architectures[arch],hidden_inputs)),          
                          ('dropout1',nn.Dropout(p=.05)),
                          ('relu1', nn.ReLU()),
                          ('fc2',nn.Linear(hidden_inputs,1024)),          
                          ('dropout2',nn.Dropout(p=.05)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(1024,102)),
                          ('output', nn.LogSoftmax(dim=1))]))
     
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),learning_rate)
    model.to(device)
    
    return model,criterion,optimizer
    
def train(model,dataloaders,criterion,optimizer,device,epochs):
    steps=0
    running_loss=0
    print_every=40

    start = time.time()
    print("**Training Started**")

    for e in range(epochs):
        for data in dataloaders[0]:
            steps += 1
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                 model.eval()
                 test_loss=0
                 accuracy=0
            
                 for data in dataloaders[1]:
                     images, labels = data
                     images, labels = images.to('cuda'), labels.to('cuda')
                     with torch.no_grad():
                            logps = model(images)
                            test_loss = criterion(logps , labels)
                            ps = torch.exp(logps)
                            top_ps, top_class = ps.topk(1, dim = 1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
           
        
                 print("Epoch: {}/{}  ".format(e+1, epochs),
                       "Training Loss: {:.3f}  ".format(running_loss/print_every),
                       "Test Loss: {:.3f}  ".format(test_loss/len(dataloaders[1])),
                       "Accuracy: {:.3f}  ".format(accuracy/len(dataloaders[1]))),
            
                 running_loss = 0
                 model.train()
            
    print("**Training Done**")
    end = time.time() - start
    print("Total time consumed in training : {:.0f}m {:.0f}s".format(end//60,end%60))
    
def accuracy(model,dataloaders,device):
    model.to(device)
    total_steps=0
    accuracy=0
    with torch.no_grad():
        for data in dataloaders[2]:
            images,labels = data
            images,labels = images.to('cuda'),labels.to('cuda')
            log_ps = model(images)
            _,ps = torch.max(log_ps.data,1)
            accuracy += sum(ps == labels).item()
            total_steps += labels.size(0)
        
    print("\nAccuracy of your model is:{:2f}\n".format(100 * (accuracy/total_steps)))
        
def load_checkpoint(path='checkpoint_01.pth'):
    checkpoint = torch.load(path)
    model = getattr(torchvision.models,'vgg16')(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
        
    return model


def main():
    print("\t DEEP NEURAL NETWORK ON FLOWER DATASET")
    arguments = arguments_parser()
    
    learning_rate = arguments.lr
    epochs = arguments.epo
    arch = arguments.arch
    gpu = arguments.gpu
    hidden_inputs = arguments.hidden_inputs
    dropout = arguments.dropout
    
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
    device = torch.device("cuda" if torch.cuda.is_available() and gpu is 'gpu' else "cpu")
    architectures= {"vgg16": 25088, "densenet121":1024}
    model,criterion,optimizer=network(arch,architectures,learning_rate,device,dropout,hidden_inputs)      
    train(model,dataloaders,criterion,optimizer,device,epochs)
    accuracy(model,dataloaders,device)
    path = arguments.save_dir
    model.class_to_idx = image_datasets[0].class_to_idx
    checkpoint = {'arch':'arch',
                  'lr':learning_rate,
                  'hl':hidden_inputs,
                  'classifier': model.classifier,
                  'dropout':dropout,
                  'epochs':epochs,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer}
    torch.save(checkpoint, path)
    print("Your model is saved at {}".format(path))
    
    
    
if __name__ == "__main__":
    main()


    