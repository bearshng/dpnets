from collections import namedtuple
from models.combinations import *
Params = namedtuple('Params', ['in_channels', 'channels', 'num_half_layer','unfolding','bn','conv_num'])
from ops.utils import soft_threshold,sparsity,kronecker,Init_DCT

class DPNetS(nn.Module):
    def __init__(self, params: Params):
        super(DPNetS, self).__init__()
        self.params=params
        in_channels=params.in_channels
        channels=params.channels
        self.num_half_layer=params.num_half_layer

        layers = []
        layers.append(Conv3dReLU(in_channels, channels, bn=params.bn))
        for _ in range(1, params.conv_num):
            layers.append(Conv3dReLU(channels, channels * 2, bn=params.bn))
            channels *= 2
        self.feature_extractor = nn.Sequential(*layers)
        dlayers = []
        wlayers = []
        # dlayers.append(nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        for _ in range(0, params.conv_num - 1):
            dlayers.append(DeConv3dReLU(channels, channels // 2, bn=params.bn))
            wlayers.append(DeConv3dReLU(channels, channels // 2, bn=params.bn))
            channels //= 2
        dlayers.append(DeConv3dReLU(channels, in_channels, bn=params.bn))
        wlayers.append(DeConv3dReLU(channels, in_channels, bn=params.bn))

        self.D = nn.Sequential(*dlayers)
        self.W = nn.Sequential(*wlayers)

        self.lam= nn.ParameterList(
            [nn.Parameter(torch.zeros(1,channels * (2**(params.conv_num-1)),1, 1,1)) for _ in range(params.unfolding)])
        [nn.init.constant_(x, 0.02) for x in self.lam]

        self.thresh_fn = soft_threshold

    def forward(self, x):
        # self.net(I.unsqueeze(1)).squeeze(1)
        # x = x.unsqueeze(1)
        sc = self.thresh_fn(self.feature_extractor(x), self.lam[0])
        # out = []
        # out.append(self.W(sc).squeeze(1))
        for i in range(1, self.params.unfolding):
            sc = self.thresh_fn(sc + self.feature_extractor(x - self.D(sc)), self.lam[i])
        out=(self.W(sc))
        return out





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
            encoder_layer = Conv3dReLU(channels, 2 * channels, k=3, s=(1, 2, 2), p=1)
            channels *= 2
            self.encoder.append(encoder_layer)
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(1, num_half_layer + 1):
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
        return out
