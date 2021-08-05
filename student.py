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
   1. choice of architecture, algorithms and enhancements (if any)
        In assignment 2, Xianchao Lin(z5276886) and Zhenshan Wei(z5286602) use residual neural network (ResNet), 
        which is a subcategory of CNN(Convolutional Neural Network). 

        The first two layers of ResNet are the same as those in GoogLeNet: a 7×7 convolutional layer with 64 output 
        channels and a stride of 2 is followed by a 3×3 maximum pooling layer with a stride of 2. The difference is 
        the batch normalization layer added after each convolutional layer of ResNet. 

        ResNet uses 4 stages composed of residual blocks (Tips 1), and each stage uses two residual blocks with the 
        same number of output channels. The number of channels of the first stage is the same as the number of input 
        channels. Since the maximum pooling layer with a stride of 2 has been used before, there is no need to reduce 
        the height and width. Each subsequent stage doubles the number of channels of the previous stage in the first 
        residual block and halves the height and width. 

        Then we add all residual blocks to ResNet. Here each stage uses 2 residual blocks.

        Finally, add the global average pooling layer and then connect the output of the fully connected layer.
        
        There are 4 convolutional layers in each stage (not counting 1×1 convolutional layers), plus the first 
        convolutional layer and the last fully connected layer, for a total of 18 layers. This model is also commonly 
        referred to as ResNet-18. 
        
        Tips 1: ResNet follows the design of VGG full 3×3 convolutional layer. First, there are 2 3×3 convolutional 
        layers with the same number of output channels in the residual block. Each convolutional layer is followed by 
        a batch normalization layer and ReLU activation function. Then we let the input skip two convolutional layers 
        and add them to the output of the convolutional layer before the ReLU activation function. Such a design 
        requires the input to have the same shape as the output of the two convolutional layers so that they can be 
        added. If you want to change the number of channels, you need to introduce an additional 1×1 convolutional 
        layer to transform the input into the required shape before doing the addition operation. 
        
        And the enhancements we develop is: (1) Do weight initialization processing (2) Replace the 7×7 large 
        convolution kernels in the input part of ResNet with three 3×3 convolution kernels. By using a large number 
        of small convolution kernels, the amount of calculation is effectively reduced (3) Put the down sampling part 
        of conv 1x1 to avgPool to avoid information loss due to the simultaneous occurrence of 1*1 convolution and 
        stride 
        



    2. choice of loss function and optimizer
        As for loss function, we choose nn.CrossEntropyLoss() and we choose optim.Adam() as our optimizer,.



    3. choice of image transformations
        The image transformations chosen by our group is different on the training set and the validation set:
        In training set we did:
        (1) transforms.Resize((224, 224))
        (2) transforms.RandomCrop(224, padding=4)
        (3) transforms.RandomHorizontalFlip()
        (4) transforms.ToTensor()
        (5) transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        
        In validation set we did:
        (1) transforms.Resize((224, 224))
        (2) transforms.ToTensor(),
        (3) transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    
    4. tuning of metaparameters
        In the final submit version of student.py, our group modify the original metaparameters and change them to:
        (1) train_val_split = 0.85; (2) batch_size = 128; (3) epochs = 200; (4) lr=0.0006
        
        Other trying of metaparameters: 

        (1) epoch: we’ve tried 20-300 epochs in same model and different models, then we found the test accuracy was 
        lower when close to 300, we think overfitting happened. Finally, we decide to set 200 epochs. 
        
        (2) learning rate: The learning rate was set from 0.0002 0.0004 0.0006 0.0008, etc. We found that when the 
        learning rate is relatively low, the training speed is too slow and model can’t converge; and if the learning 
        rate is too high, the accuracy will oscillate, the final training effect is not good. Finally, we decide to 
        set learning rate= 0.0006. 
        
        (3) train_val_split: when we choose too high ratio, overfitting occur; And if we choose too low ratio, 
        the training set is too small, which lead to underfitting. Finally, we decide to set train_val_split = 0.85. 
    
    5. use of validation set, and any other steps taken to improve generalization and avoid overfitting
    Our group use these methods to improve generalization and avoid overfitting:
        (1) Using Data Augmentation to the data set
        (2) Using batch normalization layer after convolutional layer


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
    # Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize the training set
    # so that we can get better trained model
    transform_train_list = [transforms.Resize((224, 224)),
                            transforms.RandomCrop(224, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]
    # Resize, ToTensor, Normalize the test set
    transform_test_list = [transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]

    if mode == 'train':
        return transforms.Compose(transform_train_list)
    elif mode == 'test':
        return transforms.Compose(transform_test_list)


# resnet network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Process the input image
        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # the first stage
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        # the second stage
        self.stage2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )

        self.stage2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )
        # the third stage
        self.stage3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )

        self.stage3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
        # the fourth stage
        self.stage4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )

        self.stage4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        # final stage and output the result
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(512, 14)

        # conv 1x1 from 64 to 128
        self.sub64_128 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        )

        # conv 1x1 from 128 to 256
        self.sub128_256 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        )

        # conv 1x1 from 256 to 512
        self.sub256_512 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        )

        self.relu = nn.ReLU(inplace=True)
        self.norm64 = nn.BatchNorm2d(64)
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t):

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


# cross entropy
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
train_val_split = 0.85
batch_size = 128
epochs = 200
optimiser = optim.Adam(net.parameters(), lr=0.0006)
