#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

"""


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    transform_list = [transforms.Resize((224, 224)), transforms.ToTensor(),
                      transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.229, 0.224, 0.225])]
    if mode == 'train':
        return transforms.Compose(transform_list)
    elif mode == 'test':
        return transforms.Compose(transform_list)


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.layer2 = nn.MaxPool2d(64, 64, kernel_size=3, stride=2, padding=1)
    def forward(self, t):
        print(t.shape)


class loss(nn.Module):
    """
    Class for creating a custom loss function, if desired.
    If you instead specify a standard loss function,
    you can remove or comment out this class.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss_function(output, target)


net = Network()
lossFunc = loss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 256
epochs = 3
optimiser = optim.Adam(net.parameters(), lr=0.001)
