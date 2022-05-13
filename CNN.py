import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, stride=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(3, 3), padding=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.pixelshuffle = nn.PixelShuffle(2)
    
    def forward(self, x):

        x = self.tanh(self.conv1(x))
        x1 = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x1))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(torch.cat([x, x1], dim=1)))
        x = self.relu(self.conv6(x))
        x = self.pixelshuffle(x)
        
        return x