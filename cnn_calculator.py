# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:41:39 2021

@author: mbadal1996
"""
# ======================================================================
# CNN Calculator v1.0
#
# Comments:
#
# The purpose of this small code is to allow the user to determine some
# parameters required in a CNN for a given input image size. These
# parameters change depending on the input image size and the architecture
# chosen. This sample CNN can be changed/enlarged but the basic idea remains.
#
# In particular it computes the input dimension (A*B*C) to the first fully
# connected (linear) layer near the end of the CNN. This can be tricky
# and time consuming to do by hand since filter size, stride length, etc.
# all affect this input dimension. But this calculator simplifies the process
# by using the built-in operations in Pytorch.
#
# After the example input image is fed into the CNN, the code outputs A,B,C 
# which form the input dimension (of the first linear layer) as product: A*B*C
#
# NOTE: The input image can have square or rectangular dimension.
#
# The user should look for A*B*C in the code below for any clarification.
#
#
# ======================================================================

# Pytorch
import torch
import torch.nn.functional as F

# Parameters required for example image 
batch_size = 1  # batch size is needed but any value works; default 1
channels = 3  # number of color channels; should match CNN input channels
height = 100  # example image height (allowed to differ from width)
width = 100  # example image width

# Create example image to feed into CNN.
# NOTE: This is just an image of random pixels
input_image = torch.rand(batch_size,channels,height,width)


# ---------------------------------
# Pytorch CNN class. This is just an example, but provides the basic
# building blocks needed for a more complicated CNN.

# NOTE: By default in Conv2d, stride=1 and padding=0 (meaning no padding) 

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=15,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=15,out_channels=10,kernel_size=5)
        #self.fc1 = torch.nn.Linear(A * B * C, 100)  # example (10 * 22 * 22, 100)
        #self.fc2 = torch.nn.Linear(100, 50)  # example intermediate linear layer
        #self.fc3 = torch.nn.Linear(50, 2)  # # example linear layer output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, A * B * C)   # for example (-1, 10 * 22 * 22)
        #x = F.relu(self.fc1(x))  # example activation
        #x = F.relu(self.fc2(x))  # example activation
        #x = self.fc3(x)  # final output
        
        return x

# ---------------------------------
# Another CNN class which is more sophisticated

#class Net(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
#        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=4)
#        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4)
#        self.conv3 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4)
#        self.conv4 = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4)
#        #self.fc1 = torch.nn.Linear(A * B * C, 100)  # example (128 * 3 * 3, 100)
#        #self.fc2 = torch.nn.Linear(100, 50)  # example intermediate linear layer
#        #self.fc3 = torch.nn.Linear(50, 2)  # # example linear layer output
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = self.pool(F.relu(self.conv3(x)))
#        x = self.pool(F.relu(self.conv4(x)))
#        #x = x.view(-1, A * B * C)   # for example (-1, 128 * 3 * 3)
#        #x = F.relu(self.fc1(x))  # example activation
#        #x = F.relu(self.fc2(x))  # example activation
#        #x = self.fc3(x)  # final output
#        
#        return x
    
# -------------------------------

# Create instance of "Net" class
model = Net()

# Input sample image into CNN to obtain intermediate output before linear layer
# This output will have the necessary dimensions for the first linear layer input
out = model(input_image)

# Code output which yields parameters A,B,C:
print(' ')
print('The input size of first linear layer is, A * B * C =', 
      out.size(1),'*',out.size(2),'*',out.size(3))

