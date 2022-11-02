import os
import numpy as np
import torch
from torch import nn
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torchvision.utils import save_image
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# Building block unit of encoder and decoder architecture
class Block(Module):
    def __init__(self, inChannels, outChannels, dropout = 0.15):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        # Applies the ordering we have set above which is conv1 -> relu -> conv2 -> relu
        out = self.conv1(inp)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return self.relu(out)


class Encoder(Module):
    def __init__(self, channels=(4,64,128,256,512,1024), dropout = 0.15):
        super().__init__()
        # Stores our encoder blocks which are supposed to overtime increase channel size
        self.encoder_blocks = ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        # Reduces spatial dimensions by factor of 2
        self.pool = MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        block_outputs = []
        for block in self.encoder_blocks:
            # pass input through encoder block
            inp = block(inp)
            inp = self.dropout(inp)
            # store output
            block_outputs.append(inp)
            # apply maxpooling to output to pass on to next block
            inp = self.pool(inp)
            inp = self.dropout(inp)
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64),  dropout=0.15):
        super().__init__()
        self.channels = channels
        self.upSample = nn.Upsample(scale_factor=2)
        # up-sampler block

        up_convs = []
        for i in range(len(channels)-1):
            up_convs.append(nn.Conv2d(channels[i], channels[i+1], 1, 1))
        self.up_convs = ModuleList(up_convs)

        # self.up_convs = ModuleList([
        #     ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)
        # ])
        # down-sampler block
        self.dec_blocks = ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, enc_features):
        for i in range(len(self.channels) - 1):
            # upsample
            inp = self.upSample(inp)
            inp = self.up_convs[i](inp)
            inp = self.dropout(inp)
            # crop features and concatenate with upsampled features
            enc_feat = self.crop(enc_features[i], inp)

            inp = torch.cat([inp, enc_feat], dim=1)
            # pass through decoder block
            inp = self.dec_blocks[i](inp)
            inp = self.dropout(inp)
        return inp

    def crop(self, enc_features, inp):
        # grab dims of inputs then crop encoder
        _, _, h, w = list(inp.shape)
        enc_features = CenterCrop(h)(enc_features)
        return enc_features

class UNet(Module):
    def __init__(self, enc_channels=(4,64,128,256,512,1024), dec_channels=(1024, 512, 256, 128, 64), num_class=1,
                 retain_dim=False, out_sz=(572,572), dropout=0.15):
        super().__init__()
        self.encoder = Encoder(enc_channels,  dropout=dropout)
        self.decoder = Decoder(dec_channels,  dropout=dropout)
        self.head = Conv2d(dec_channels[-1], num_class, 1)
        self.out_sz = out_sz
        self.retain_dim = retain_dim


    def forward(self, inp):
        enc_features = self.encoder(inp)
        out = self.decoder(enc_features[-1], enc_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz).clamp(min=0, max=1)
        return out

def create_pretrained(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=3, preprocess_only=False):
    preprocess_fn = None
    if encoder_weights:
        preprocess_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
        if preprocess_only:
            return preprocess_fn
    model = smp.Unet(
        encoder_name=encoder_name,  # resnet34 resnet50
        encoder_weights=encoder_weights,  # use 'imagenet' 'swsl'
        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=classes,  # model output channels (number of classes in your dataset)
        activation=None,  # type of activation function for the final layer
        decoder_use_batchnorm=True
    )
    return model, preprocess_fn


if __name__ == '__main__':
    rand_inp = torch.randn(1, 3, 256, 256)
    label = torch.randn(1,3,256,256)
    rand_inp_2 = torch.randn(1, 3, 512, 512)
    label2 = torch.randn(1,3,512,512)
    # unet = UNet(num_class = 3, retain_dim=True, out_sz=(256, 256))

    pass




