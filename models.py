
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,32,4)
        self.pool1=nn.MaxPool2d(2,2)
        
        self.conv2=nn.Conv2d(32,64,3)
        self.pool2=nn.MaxPool2d(2,2)
        
        self.conv3=nn.Conv2d(64,128,2)
        self.pool3=nn.MaxPool2d(2,2)
        
        self.conv4=nn.Conv2d(128,256,1)
        self.pool4=nn.MaxPool2d(2,2)
        
        
        self.fc1=nn.Linear(256*13*13,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,136)
        
        
        
        self.fc1_drop=nn.Dropout(p=0.1)
        self.fc2_drop=nn.Dropout(p=0.2)
        self.fc3_drop=nn.Dropout(p=0.3)
        self.fc4_drop=nn.Dropout(p=0.4)
        self.fc5_drop=nn.Dropout(p=0.5)
        self.fc6_drop=nn.Dropout(p=0.6)
        
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.fc1_drop(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.fc2_drop(x)
        
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.fc3_drop(x)
        
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.fc4_drop(x)
        
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc5_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc6_drop(x)
        x = self.fc3(x)

        

        return x

        

        
  