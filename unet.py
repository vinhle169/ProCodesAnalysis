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


# Building block unit of encoder and decoder architecture
# noinspection PyTypeChecker
class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, inp):
        # Applies the ordering we have set above which is conv1 -> relu -> conv2 -> reli
        return self.relu(self.conv2(self.relu(self.conv1(inp))))


class Encoder(Module):
    def __init__(self, channels=(4,64,128,256,512,1024)):
        super().__init__()
        # Stores our encoder blocks which are supposed to overtime increase channel size
        self.encoder_blocks = ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        # Reduces spatial dimensions by factor of 2
        self.pool = MaxPool2d(2)

    def forward(self, inp):
        block_outputs = []
        for block in self.encoder_blocks:
            # pass input through encoder block
            inp = block(inp)
            # store output
            block_outputs.append(inp)
            # apply maxpooling to output to pass on to next block
            inp = self.pool(inp)
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
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

    def forward(self, inp, enc_features):
        for i in range(len(self.channels) - 1):
            # upsample
            inp = self.up_convs[i](inp)
            # crop features and concatenate with upsampled features
            enc_feat = self.crop(enc_features[i], inp)
            inp = torch.cat([inp, enc_feat], dim=1)
            # pass through decoder block
            inp = self.dec_blocks[i](inp)
        return inp

    def crop(self, enc_features, inp):
        # grab dims of inputs then crop encoder
        print(enc_features.shape)
        print(inp.shape)
        _, _, h, w = list(inp.shape)
        enc_features = CenterCrop(h)(enc_features)
        return enc_features

class UNet(Module):
    def __init__(self, enc_channels=(4,64,128,256,512,1024), dec_channels=(1024, 512, 256, 128, 64), num_class=1,
                 retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
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


class FocalLoss(Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

if __name__ == '__main__':
    unet = UNet(num_class = 3, retain_dim=True)
    unet = nn.DataParallel(unet)
    x = torch.randn(2,4,244,244).cuda()
    y = torch.randn(2,3,244,244).cuda()
    unet = unet.cuda()
    print(unet(x).shape)
    # l = FocalLoss()
    # loss = (unet(x), y)
    # print(loss)