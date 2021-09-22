# python 3.8.5 
# torch 1.9.0


#import necessary libraries

import torch
from torch._C import _LegacyVariableBase
import torch.nn as nn

# creating a conv block
# This block can be used for only ResNet50, ResNet101, ResNet152 architectures

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        identity_downsample=None,
        stride=1
        ):
        super(ConvBlock, self).__init__()
        """
            Arguments: 
                    1. in_channels : int : Number of kernels at input of each block
                    2. out_channels : int : Number of kernels at output of each block
                    3. identity_downsample
                    4. stride : int
        """
        self.expansion = 4 # Ratio between the input channels and output channels
        
        self.conv_2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm_2 = nn.BatchNorm2d(out_channels) 

        self.conv_3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.batch_norm_3 = nn.BatchNorm2d(out_channels) 

        self.conv_4 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm_4 = nn.BatchNorm2d(out_channels*self.expansion) 

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    
    def forward(self, x):
        identity = x.clone()

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)

        x = self.conv_4(x)
        x = self.batch_norm_4(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, cnn_block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv_1 = nn.Conv2d(
            image_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Creating layers
        self.layer_1 = self._make_layer(
            cnn_block, layers[0], out_channels=64, stride=1
        )
        self.layer_2 = self._make_layer(
            cnn_block, layers[1], out_channels=128, stride=2
        )
        self.layer_3 = self._make_layer(
            cnn_block, layers[2], out_channels=256, stride=2
        )
        self.layer_4 = self._make_layer(
            cnn_block, layers[3], out_channels=512, stride=2
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense_layer = nn.Linear(512*4, num_classes)


    def forward(self, x):
        """
        Args : 
            x : Input Image Tensor
        Returns :
           Outputs image Tensor passed through the ResNet layers
        """
        x = self.conv_1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.max_pool(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dense_layer(x)
        
        return x


    def _make_layer(self, cnn_block, num_residual_blocks, out_channels, stride):
        """
            Creates a layer with the Cnn Block instance
        """
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * 4),
            )
        
        layers.append(
            cnn_block(self.in_channels, out_channels, identity_downsample, stride)
        )

        self.in_channels = out_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(cnn_block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(ConvBlock, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(ConvBlock, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(ConvBlock, [3, 8, 36, 3], img_channel, num_classes)


def test():
    net = ResNet101(img_channel=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224))
    print(y.size())


if __name__ == "__main__":
    test()
