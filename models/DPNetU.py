from collections import namedtuple
from models.combinations import *
Params = namedtuple('Params', ['in_channels', 'channels', 'num_half_layer','unfolding','bn','conv_num'])
import  scipy.io as scio
class DPNetU(nn.Module):
    def __init__(self, params: Params):
        super(DPNetU, self).__init__()
        self.params=params
        in_channels=params.in_channels
        channels=params.channels
        self.num_half_layer=params.num_half_layer
        self.feature_extractor = nn.Sequential(
            Conv3dReLU(in_channels, channels, k=3, s=1, p=1,bn=params.bn),
            Conv3dReLU(channels, channels * 2, k=3, s=1, p=1,bn=params.bn),
            Conv3dReLU(channels * 2, channels * 4, k=3, s=1, p=1,bn=params.bn),
        )
        self.D = nn.Sequential(
            DeConv3dReLU(channels * 4, channels * 2,bn=params.bn),
            DeConv3dReLU(channels * 2, channels,bn=params.bn),
            DeConv3dReLU(channels, in_channels,bn=params.bn),
        )

        self.W = nn.Sequential(
            DeConv3dReLU(channels * 4, channels * 2,bn=params.bn),
            DeConv3dReLU(channels * 2, channels,bn=params.bn),
            DeConv3dReLU(channels, in_channels,bn=params.bn),
        )
        self.prox=RED3D(channels * 4, channels, self.num_half_layer)

   
    def forward(self, x):
        # self.net(I.unsqueeze(1)).squeeze(1)
        #x = x.unsqueeze(1)
        sc = self.prox(self.feature_extractor(x))
        for i in range(1, self.params.unfolding):
            sc = self.prox(sc + self.feature_extractor(x - self.D(sc)))
        out = self.W(sc)
        return out


class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False,bn=False):
        super(Conv3dReLU, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))
        if bn:
            self.add_module('bn', BatchNorm3d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))

class DeConv3dReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False,bn=False):
        super(DeConv3dReLU, self).__init__()
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))
        if bn:
            self.add_module('bn', BatchNorm3d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))

class RED3D(torch.nn.Module):
    """Residual Encoder-Decoder Convolution 3D
    Args:
        downsample: downsample times, None denotes no downsample"""

    def __init__(self, in_channels, channels, num_half_layer, downsample=None):
        super(RED3D, self).__init__()
        # Encoder
        # assert downsample is None or 0 < downsample <= num_half_layer
        interval = 2

        self.feature_extractor = Conv3dReLU(in_channels, channels)
        self.encoder = nn.ModuleList()
        for i in range(1, num_half_layer + 1):
            if i % interval:
                encoder_layer = Conv3dReLU(channels, channels)
            else:
                encoder_layer = Conv3dReLU(channels, 2 * channels, k=3, s=(1, 2, 2), p=1)
                channels *= 2
            self.encoder.append(encoder_layer)
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(1, num_half_layer + 1):
            if i % interval:
                decoder_layer = DeConv3dReLU(channels, channels)
            else:
                decoder_layer = UpsampleConv3dReLU(channels, channels // 2)
                channels //= 2
            self.decoder.append(decoder_layer)
        self.reconstructor = DeConv3dReLU(channels, in_channels)
        # self.enl_1 = EfficientNL(in_channels=channels)

     # = None, head_count = None,  = None
    def forward(self, x):
        num_half_layer = len(self.encoder)
        xs = [x]
        out = self.feature_extractor(xs[0])
        xs.append(out)
        if num_half_layer % 2 != 0:
            for i in range(num_half_layer - 1):
                out = self.encoder[i](out)
                xs.append(out)
            out = self.encoder[-1](out)
            # out = self.nl_1(out)
            out = self.decoder[0](out)
            for i in range(1, num_half_layer):
                out = out + xs.pop()
                out = self.decoder[i](out)
            out = (out) + xs.pop()
            out = self.reconstructor(out)
            out = (out) + xs.pop()
        else:
            out = self.encoder[0](out)
            for i in range(1, num_half_layer):
                out = self.encoder[i](out)
                xs.append(out)
                # print(out.shape)
            # out = self.encoder[-1](out)
            # out = self.nl_1(out)
            out = self.decoder[0](out)
            for i in range(1, num_half_layer):
                # print(out.shape)
                out = out + xs.pop()
                out = self.decoder[i](out)
            # print(out.shape)
            out = (out) + xs.pop()
            out = self.reconstructor(out)
            # print(out.shape)
            temp=xs.pop()
            # print(temp.shape)
            out = (out) +temp
        return out