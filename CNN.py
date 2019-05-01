__author__ = "Aisha Urooj"

import torch.nn as nn
import torch
import math

class Model_1(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()

        # here we initialize both halves of our fully connected layer 
        # and its activation function
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):

        # we pass our data through the first half and then through the activation so that
        # the neurons can be fired off correctly which maps all our date through to the output layer
        # the output layer has 10 classes which it maps all the data to 
        features = self.input_layer(x)
        features = self.sigmoid(features)
        features = self.output_layer(features)
        
        return features

class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # here we add an extra convolution layer to our fully connected one
        # this greatly increases the cost of processing and has much more
        # calculations to do 
        self.convolve1 = nn.Conv2d(1, 40, kernel_size = 5, stride = 1, padding = 2)

        # we pool our data in order to select the maximum from each kernel 
        # this cuts our size down by half 
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.batch = nn.BatchNorm2d(40)

        self.convolve2 = nn.Conv2d(40, 40, kernel_size = 5, stride = 1, padding = 2)

        # we pool our data in order to select the maximum from each kernel 
        # this cuts our size down by half 
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.sigmoid = nn.Sigmoid()

        # the dimensions are 7, 7, 40 because we have 40 output channels
        # from the convolution and our image has been cut down by 1/4 in dimension 
        self.fc1 = nn.Linear(7 * 7 * 40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        inSize = x.size(0)

        # we pass our data through our convoluion layers
        features = self.convolve1(x)
        features = self.batch(features)

        # each layer needs to be activated before its passed onto 
        # the next layer
        features = self.sigmoid(features)
        features = self.pool1(features)

        features = self.convolve2(features)
        features = self.batch(features)
        features = self.sigmoid(features)
        features = self.pool2(features)

        # here we resize our tensor to be flattened
        # this lets us pass through our convolution layer
        features = features.view(inSize, -1)

        features = self.fc1(features)
        features = self.sigmoid(features)
        features = self.fc2(features)

        return features

class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # This model is the same as model 2 except between the convolution layers
        # we do ReLU activation instead 
        self.convolve1 = nn.Conv2d(1, 40, kernel_size = 5, stride = 1, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.convolve2 = nn.Conv2d(40, 40, kernel_size = 5, stride = 1, padding = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(7 * 7 * 40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        inSize = x.size(0)
        
        # we pass our data through our convoluion layers
        # each layer needs to be activated before its passed onto 
        # the next layer
        features = self.convolve1(x)
        features = self.relu(features)
        features = self.pool1(features)


        features = self.convolve2(features)
        features = self.relu(features)
        features = self.pool2(features)

        #  here we resize our tensor to be flattened
        # this lets us pass through our convolution layer
        features = features.view(inSize, -1)

        features = self.fc1(features)
        features = self.sigmoid(features)
        features = self.fc2(features)

        return features 

class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # In this model we added an extra fully connected layer to have 2 convolutions and 2 fully connected layers
        # in our model.  All other paramters remain the same
        self.convolve1 = nn.Conv2d(1, 40, kernel_size = 5, stride = 1, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.convolve2 = nn.Conv2d(40, 40, kernel_size = 5, stride = 1, padding = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(7 * 7 * 40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10) 

    def forward(self, x):
        inSize = x.size(0)
        features = self.convolve1(x)
        features = self.relu(features)
        features = self.pool1(features)

        features = self.convolve2(features)
        features = self.relu(features)
        features = self.pool2(features)

        features = features.view(inSize, -1)

        features = self.fc1(features)
        features = self.sigmoid(features)
        features = self.fc2(features)
        features = self.sigmoid(features)
        features = self.fc3(features)

        return features

class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.convolve1 = nn.Conv2d(1, 40, kernel_size = 5, stride = 1, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.convolve2 = nn.Conv2d(40, 40, kernel_size = 5, stride = 1, padding = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(7 * 7 * 40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        inSize = x.size(0)
        
        features = self.convolve1(x)
        features = self.relu(features)
        features = self.pool1(features)

        features = self.convolve2(features)
        features = self.relu(features)
        features = self.pool2(features)
        
        features = features.view(inSize, -1)
        features = self.dropout(features)
        
        features = self.fc1(features)
        features = self.sigmoid(features)
        features = self.fc2(features)
        features = self.sigmoid(features)
        features = self.fc3(features)

        return features


class Net(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.hidden_size= args.hidden_size
        # model 1: base line
        if mode == 1:
            in_dim = 28*28 # input image size is 28x28
            self.model = Model_1(in_dim, self.hidden_size)

        # model 2: use two convolutional layer
        if mode == 2:
            self.model = Model_2(self.hidden_size)

        # model 3: replace sigmoid with relu
        if mode == 3:
            self.model = Model_3(self.hidden_size)

        # model 4: add one extra fully connected layer
        if mode == 4:
            self.model = Model_4(self.hidden_size)

        # model 5: utilize dropout
        if mode == 5:
            self.model = Model_5(self.hidden_size)


    def forward(self, x):
        if self.mode == 1:
            x = x.view(-1, 28* 28)
            x = self.model(x)
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)
    
        # logits = nn.functional.softmax(x, dim = 1)
        logits = nn.functional.log_softmax(x, dim = 0)
        return logits

