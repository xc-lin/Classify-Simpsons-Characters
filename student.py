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
from torchvision.models import resnet18

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
        self.layer1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer11 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer14 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer15 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.layer16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer18 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer19 = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(512, 14)

        self.sub1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        self.sub2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.sub3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)

    def forward(self, t):
        layer1 = self.layer1(t)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4 + layer2)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6 + layer4 + layer2)
        layer8 = self.layer8(layer7)
        layer_sub1 = self.sub1(layer6 + layer4 + layer2)
        layer9 = self.layer9(layer8 + layer_sub1)
        layer10 = self.layer10(layer9)
        layer11 = self.layer11(layer10 + layer8 + layer_sub1)
        layer12 = self.layer12(layer11)
        layer_sub2 = self.sub2(layer10 + layer8 + layer_sub1)
        layer13 = self.layer13(layer12 + layer_sub2)
        layer14 = self.layer14(layer13)
        layer15 = self.layer15(layer14 + layer12 + layer_sub2)
        layer16 = self.layer16(layer15)
        layer_sub3 = self.sub3(layer14 + layer12 + layer_sub2)
        layer17 = self.layer17(layer16 + layer_sub3)
        layer18 = self.layer18(layer17)
        layer19 = self.layer19(layer18 + layer16 + layer_sub3)
        layer19_flattened = torch.flatten(layer19, 1)
        output = self.output(layer19_flattened)
        return output


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
