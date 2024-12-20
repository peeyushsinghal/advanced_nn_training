import torch
import torch.nn as nn  # for network
import torch.nn.functional as F  # for forward method

drop_out_value = 0.1


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()  # extending super class method

        # Input block
        self.convblock_input = nn.Sequential(
            nn.Conv2d(
                3, 32, 3, padding=1
            ),  # In- 3x32x32 (CxHxW), Out- 32x32x32, RF- 3x3, Jump_in-1, Jump_out-1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
        )

        # CONV BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                32, 32, 3, padding=1
            ),  # In- 32x32x32, Out- 32x32x32, RF- 5x5, Jump_in -1, Jump_out -1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
            nn.Conv2d(
                32, 32, 3, padding=1
            ),  # In- 32x32x32, Out- 32x32x32, RF- 7x7, Jump_in -1, Jump_out -1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
        )

        # TRANSITION BLOCK 1
        # STRIDED CONVOLUTION LAYER
        self.transitionblock1 = nn.Sequential(
            nn.Conv2d(
                32, 32, 3, stride=2, padding=1
            ),  # In- 32x32x32, Out- 32x16x16, RF- 9x9, Jump_in -1, Jump_out -2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
        )

        # CONV BLOCK 2
        # Depthwise Separable Convolution Layer
        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                32, 32, 3, padding=1, groups=32
            ),  # In- 32x16x16, Out- 32x16x16, RF- 13x13, Jump_in -2, Jump_out -2
            nn.Conv2d(
                32, 32, 1, padding=0
            ),  # In-32x16x16 , Out- 32x16x16, RF- 13x13, Jump_in -2, Jump_out -2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
            nn.Conv2d(
                32, 32, 3, padding=1
            ),  # In-32x16x16 , Out-32x16x16 , RF- 17x17, Jump_in -2, Jump_out -2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
        )

        # TRANSITION BLOCK 2
        # STRIDED CONVOLUTION LAYER
        self.transitionblock2 = nn.Sequential(
            nn.Conv2d(
                32, 32, 3, stride=2, padding=1
            ),  # In- 32x16x16, Out-32x8x8 , RF- 21x21, Jump_in -2, Jump_out -4
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
        )

        # CONV BLOCK 3
        # Dilated Convolution Layer
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                32, 32, 3, padding=1, dilation=2
            ),  # In- 32x8x8, Out-32x6x6 , RF- 29x29, Jump_in -4, Jump_out -4
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
            nn.Conv2d(
                32, 32, 3, padding=1
            ),  # In-32x6x6 , Out- 32x6x6, RF- 37x37, Jump_in -4, Jump_out -4
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
        )

        # TRANSITION BLOCK 3
        # STRIDED CONVOLUTION LAYER
        self.transitionblock3 = nn.Sequential(
            nn.Conv2d(
                32, 32, 3, stride=2, padding=1
            ),  # In-32x6x6 , Out-32x3x3 , RF- 45x45, Jump_in -4, Jump_out -8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
        )

        # CONV BLOCK 4
        # Depthwise Separable Convolution Layer
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                32, 32, 3, padding=1
            ),  # In- 32x3x3, Out-32x3x3 , RF- 61x61, Jump_in -8, Jump_out -8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(drop_out_value),
            nn.Conv2d(
                32, 32, 3, padding=1, groups=32
            ),  # In-32x3x3 , Out-32x3x3 , RF- 77x77, Jump_in -8, Jump_out -8
            nn.Conv2d(
                32, 10, 1, padding=0
            ),  # In- 32x3x3, Out-10x3x3 , RF- 77x77, Jump_in -8, Jump_out -8
        )

        # Output BLOCK
        # GAP Layer
        self.gap = nn.AvgPool2d(
            3
        )  # In- 10x3x3, Out-10x1x1 , RF- 77x77, Jump_in -8, Jump_out -8

    def forward(self, x):
        x = self.convblock_input(x)
        x = self.convblock1(x)
        x = self.transitionblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock2(x)
        x = self.convblock3(x)
        x = self.transitionblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
