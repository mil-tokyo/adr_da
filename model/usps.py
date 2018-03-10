import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data_utils
from utils import *

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5,stride=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5,stride=1)
        self.bn2 = nn.BatchNorm2d(50)        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2,kernel_size=2,dilation=(1,1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2,kernel_size=2,dilation=(1,1))
        x = x.view(x.size(0), 800)
        return x

class Predictor(nn.Module):
    def __init__(self,prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(800, 500)
        self.bn1_fc = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)
        self.bn2_fc = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(500,10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob=prob
    def forward(self, x):
        x = F.dropout(x, training=self.training,p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training,p=self.prob)
        x = self.fc2(x)
        return x

