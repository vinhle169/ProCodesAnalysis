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

# Building block unit of encoder and decoder architecture
# noinspection PyTypeChecker
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
        # up-sampler block
        self.up_convs = ModuleList([
            ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)
        ])
        # down-sampler block
        self.dec_blocks = ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, enc_features):
        for i in range(len(self.channels) - 1):
            # upsample
            inp = self.up_convs[i](inp)
            inp = self.dropout(inp)
            # crop features and concatenate with upsampled features
            enc_feat = self.crop(enc_features[i], inp)
            enc_feat = self.dropout(enc_feat)
            inp = torch.cat([inp, enc_feat], dim=1)
            inp = self.dropout(inp)
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
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


def to_one_hot(tensor, nClasses):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=3, reduction='mean'):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot, reduction='mean'):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union
        ## Return average loss over classes and batch
        return 1-loss.mean()

if __name__ == '__main__':
    directory = 'models/models/'
    test_images = ['/nobackup/users/vinhle/data/blobs/F051_trim_manual_9.pt', '/nobackup/users/vinhle/data/blobs/F030_trim_manual_4.pt']
    for model_name in os.listdir(directory):
        print(f'Trying {model_name}')
        checkpoint = torch.load(directory+model_name)
        if 'dropout' in model_name:
            unet = UNet(num_class = 3, retain_dim=True, out_sz = (2048,2048))
        else:
            unet = UNet(num_class = 3, retain_dim=True, out_sz = (2048,2048), dropout=0)
        unet = nn.DataParallel(unet)
        unet.load_state_dict(checkpoint['model_state_dict'])
        unet.eval()
        cuda0 = torch.device('cuda:0')
        unet.to(cuda0)
        for i in range(len(test_images)):
            img = torch.load(test_images[i]).to(cuda0)
            img = img.view((1,4,2048,2048))
            output = unet(img)
            torch.save(output, f'{model_name[:-3]}_{i}.pt')

        del unet
        del checkpoint


