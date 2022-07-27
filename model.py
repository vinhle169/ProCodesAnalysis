# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time

# check if gpu is available
use_gpu = torch.cuda.is_available()
print("Is gpu available:", use_gpu)

# noinspection PyTypeChecker
class ColorizationNet(nn.Module):
    def __init__(self, input_size=2048):
        super(ColorizationNet, self).__init__()

        self.model1 = nn.Sequential(*
            [
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(64)
            ]
        )

        self.model2 = nn.Sequential(*
            [
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(128)
            ]
        )

        self.model3 = nn.Sequential(*
            [
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(256)
            ]
        )

        self.model4 = nn.Sequential(*
            [
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512)
            ]
        )

        self.model5 = nn.Sequential(*
            [
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512)
            ]
        )

        self.model6 = nn.Sequential(*
            [
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512)
            ]
        )

        self.model7 = nn.Sequential(*
            [
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512)
            ]
        )

        self.model8 = nn.Sequential(*
            [
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=0)
            ]
        )

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, input0):
        # Pass in inputs through each of the sets of conv layers, input should be normalized
        conv_01 = self.model1(input0)
        conv_12 = self.model2(conv_01)
        conv_23 = self.model3(conv_12)
        conv_34 = self.model4(conv_23)
        conv_45 = self.model5(conv_34)
        conv_56 = self.model6(conv_45)
        conv_67 = self.model7(conv_56)
        conv_78 = self.model8(conv_67)
        out_8 = self.model_out(self.softmax(conv_78))
        return self.upsample4(out_8)

if __name__ == "__main__":
    learning_rate = 0.001
    model = ColorizationNet()
    criterion = nn.MSELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
    # total_steps = 4
