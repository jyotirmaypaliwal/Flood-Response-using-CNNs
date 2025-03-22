# For data handling
import numpy as np
#import pandas as pd

# For plotting dataset.
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import seaborn as sns

# Other useful libraries
import os
import sys

# Main library
import torch 

# For image processing
import torchvision 
import torchvision.transforms as transforms 
from torchvision.io import read_image

# For data handling and batch processing
from torch.utils.data import Dataset ,DataLoader

#Others
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

#%matplotlib inline




# Image transformation
transformations_v1 = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])




# Creating dataset classs
dataset = torchvision.datasets.ImageFolder("C:\\Users\\jyoti\\Desktop\\Utkarshini\\classified_images\\", transform = transformations_v1)
dataset_test = torchvision.datasets.ImageFolder("C:\\Users\\jyoti\\Desktop\\Utkarshini\\classified_images_test\\", transform = transformations_v1)



# Data loader
train_dataloader = DataLoader(dataset, batch_size=100, shuffle=True )
test_dataloader = DataLoader(dataset_test, batch_size=100, shuffle=True )





## Setting up our device(cpu or gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing and customizing model
from torchvision import models
model = models.inception_v3(pretrained=True)
for param in model.parameters():
    param.requires_grad=False
    
model.AuxLogits.fc = nn.Sequential(nn.Linear(768, 1),
                                   nn.Sigmoid())
model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1),
                        nn.Sigmoid())


# More device configurations
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device);






import torch.optim 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def compute_loss(y_hat, y):
    return F.binary_cross_entropy(y_hat, y)




for epoch in range(20):
    total_loss = 0
    
    for batch in train_dataloader:
        image, label = batch
        label = torch.unsqueeze(label,1).float()

        image = image.to(device=device)
        label = label.to(device=device)

        
        outputs, aux_outputs = model(image)
        loss = compute_loss(outputs, label) + (0.4 * (compute_loss(aux_outputs, label)))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print("epoch: ", epoch, " loss: ", total_loss)




tl=0
total_correct=0
for batch in test_dataloader:
    image, label = batch
    label = torch.unsqueeze(label,1).float()
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.to(device);
    image = image.to(device=device)
    label = label.to(device=device)
    
    outputs, aux_outputs = model(image)
    loss = compute_loss(outputs, label) 
    
    tl += loss.item()
    
    
    
    l = np.array([[]])
    for output in outputs:
        if output >= 0.5:
            output =1
            l=np.append(l,output)
        else:
            output=0
            l=np.append(l,output)
            
    l = torch.from_numpy(l)
    l = torch.unsqueeze(l,1).float().to(device=device)
    
      
    
    
    def correct_pred(output, label):
        return output.eq(label).sum().item()
    
    
    
    total_correct += correct_pred(l, label)
    
    
    
print("Total loss: ", tl, " Total correct: ", total_correct, " 5000")



accuracy = ((total_correct*100)/5000)
print("Our accuracy percent is: ", accuracy)
    
