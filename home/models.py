## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from collections import OrderedDict

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) 
        #to avoid overfitting
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=11, stride=4, padding=2)),
            ('relu', nn.ReLU(inplace=True)),
            # 55 * 55
            ('max1', nn.MaxPool2d(kernel_size=3, stride=2)),
            #27 * 27
            ('conv2', nn.Conv2d(32, 64, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU(inplace=True)),
            # 12 * 12
            ('max2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(64, 192, kernel_size=3, padding=1)),
            # 6 * 6
            ('relu3', nn.ReLU(inplace=True))
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout()),
            ('fc1', nn.Linear(192 * 6 * 6, 256)),
            ('relu4',nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(256, 136)),
        ]))
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
