# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn





class UNET(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.L_ReLu = nn.LeakyReLU(0.2, inplace = True)
        
        
        #convolutions
        self.conv1 = nn.Conv2d(3,16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(32,32,kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(64,64,kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(128,128, kernel_size = 3, padding = 1)
        self.conv9 = nn.Conv2d(68,64,kernel_size = 3, padding = 1)
        self.conv10 = nn.Conv2d(36,32,kernel_size = 3, padding = 1)
        self.conv12 = nn.Conv2d(128,64, kernel_size = 3, padding = 1)
        self.conv13 = nn.Conv2d(64,32, kernel_size = 3, padding = 1)
        self.conv14 = nn.Conv2d(32,16, kernel_size = 3, padding = 1)
        
        
        #down convolution
        self.conv2 = nn.Conv2d(16,32,kernel_size = 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32,64,kernel_size = 3, stride = 2, padding = 1)
        self.conv6 = nn.Conv2d(64,64, kernel_size = 3, stride = 2, padding = 1)
        self.conv7 = nn.Conv2d(64,128, kernel_size = 3, stride = 2, padding = 1)
        
        #Ultima convolução
        self.conv11 = nn.Conv2d(20,3, kernel_size = 1)
        
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        
        #batch normalization Skip connection
        self.bnS = nn.BatchNorm2d(4)
        
        #skip convolution
        self.skip1 = nn.Conv2d(16,4,kernel_size = 1)
        self.skip2 = nn.Conv2d(32,4,kernel_size = 1)
        self.skip3 = nn.Conv2d(64,4,kernel_size = 1)
        
        self.Up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        
        self.concat= Concat()
        
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        
        #Contract path
        y = self.L_ReLu(self.bn1(self.conv1(x)))
        s1 = self.L_ReLu(self.bnS(self.skip1(y)))
        
        y = self.L_ReLu(self.bn2(self.conv2(y)))
        y = self.L_ReLu(self.bn2(self.conv3(y)))
        s2 = self.L_ReLu(self.bnS(self.skip2(y)))
        
        y = self.L_ReLu(self.bn3(self.conv4(y)))
        y = self.L_ReLu(self.bn3(self.conv5(y)))
        s3 = self.L_ReLu(self.bnS(self.skip3(y)))
        
        
        y = self.L_ReLu(self.bn3(self.conv6(y)))
        y = self.L_ReLu(self.bn3(self.conv5(y)))
        s4 = self.L_ReLu(self.bnS(self.skip3(y)))
        
        
        #Middle
        y = self.L_ReLu(self.bn4(self.conv7(y)))
        y = self.L_ReLu(self.bn4(self.conv8(y)))
        
        
        
        #Expensive path
        y = self.L_ReLu(self.bn3(self.conv12(self.Up(y))))
       
        y1 = self.concat(y , s4)
        y = self.L_ReLu(self.bn3(self.conv9(y1)))
        
        y = self.L_ReLu(self.bn3(self.conv5(self.Up(y))))
       
        y1 = self.concat(y , s3)
        y = self.L_ReLu(self.bn3(self.conv9(y1)))
        
        y = self.L_ReLu(self.bn2(self.conv13(self.Up(y))))

        
        y1 = self.concat(y , s2)
        y = self.L_ReLu(self.bn2(self.conv10(y1)))
        
        y = self.L_ReLu(self.bn1(self.conv14(self.Up(y))))

        y1 = self.concat(y , s1)
        y = self.output(self.conv11(y1))
        
        return y
    
    
    
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)    