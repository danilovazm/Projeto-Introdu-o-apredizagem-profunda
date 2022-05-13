import os
import numpy as np
import torch.nn as nn

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.conv4.weight, mean=0.0, std=0.01)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.conv5.weight, mean=0.0, std=0.01)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.tconv1.weight, mean=0.0, std=0.01)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.tconv2.weight, mean=0.0, std=0.01)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.tconv3.weight, mean=0.0, std=0.01)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.tconv4.weight, mean=0.0, std=0.01)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=5, stride=1, padding=0)
        nn.init.normal_(self.tconv5.weight, mean=0.0, std=0.01)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out
