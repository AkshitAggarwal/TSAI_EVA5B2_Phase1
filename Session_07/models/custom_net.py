import torch
import torch.nn as nn
import torch.nn.functional as F

def convBlock(in_channels, out_channels, kernel_size):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding = 1, bias = False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(),
                         nn.Dropout(0.10)
                         )
                
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Convolution block: 1
        self.conv1 = nn.Sequential(convBlock(in_channels = 3, out_channels = 32, kernel_size = 3), #32x32x3 -> 32x32x32
                                    convBlock(32, 32, 3), #32x32x32 -> 32x32x32
                                    convBlock(32, 32, 3) #32x32x32 -> 32x32x32
                                    )

        #Transition block: 1
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2) #32x32x32 -> 16x16x32

        #Convolution block: 2
        self.conv2 = nn.Sequential(convBlock(32, 64, 3), #16x16x32 -> 16x16x64
                                    convBlock(64, 64, 3), #16x16x64 -> 16x16x64
                                    convBlock(64, 64, 3) #16x16x64 -> 16x16x64
                                    )

        #Transition block: 2
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2) #16x16x64 -> 8x8x64

        #Convolution block: 3
        self.conv3 = nn.Sequential(convBlock(64, 128, 3), #8x8x64 -> 8x8x128
                                    convBlock(128, 128, 3), #8x8x128 -> 8x8x128
                                    convBlock(128, 128, 3) #8x8x128 -> 8x8x128
                                    )

        #Transition block: 3
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2) #8x8x128 -> 4x4x128

        #Convolution block: 4
        self.conv4 = nn.Sequential(convBlock(128, 128, 3), #4x4x128 -> 4x4x128
                                    convBlock(128, 128, 3), #4x4x128 -> 4x4x128
                                    convBlock(128, 128, 3) #4x4x128 -> 4x4x128
                                    )

        #GAP Layer
        self.gap = nn.AvgPool2d((4, 4)) #4x4x128 -> 1x1x128
        self.FC = nn.Conv2d(128, 10, 1) #1x1x128 -> 1x1x10

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.FC(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim = -1)