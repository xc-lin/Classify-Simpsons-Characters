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
        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.stage2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.stage2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.stage3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.stage3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.stage4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )

        self.stage4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(512, 14)

        self.sub64_128 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        self.sub128_256 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.sub256_512 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.norm64 = nn.BatchNorm2d(64)

    def forward(self, t):
        # return self.resnet18.forward(t)
        init = self.init(t)

        x = self.stage1(init)
        stage1_1_out = self.relu(x + init)

        x = self.stage1(stage1_1_out)
        stage1_2_out = self.relu(x + stage1_1_out)

        x = self.stage2_1(stage1_2_out)
        stage1_2_out_down = self.sub64_128(stage1_2_out)
        stage2_1_out = self.relu(stage1_2_out_down + x)

        x = self.stage2_2(stage2_1_out)
        stage2_2_out = self.relu(stage2_1_out + x)

        x = self.stage3_1(stage2_2_out)
        stage2_2_out_down = self.sub128_256(stage2_2_out)
        stage3_1_out = self.relu(stage2_2_out_down + x)

        x = self.stage3_2(stage3_1_out)
        stage3_2_out = self.relu(stage3_1_out + x)

        x = self.stage4_1(stage3_2_out)
        stage3_2_out_down = self.sub256_512(stage3_2_out)
        stage4_1_out = self.relu(stage3_2_out_down + x)

        x = self.stage4_2(stage4_1_out)
        stage4_2_out = self.relu(stage4_1_out + x)
        avg = self.avg_pool(stage4_2_out)

        flatten = torch.flatten(avg, 1)
        output = self.output_layer(flatten)

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
epochs = 20
optimiser = optim.Adam(net.parameters(), lr=0.001)
