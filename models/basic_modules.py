import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.DeformableBlock3D import DeformConv3d, AttDeformConv3d

norm_dict = {'BATCH': nn.BatchNorm3d, 'INSTANCE': nn.InstanceNorm3d, 'GROUP': nn.GroupNorm}


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True, deform=False):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        self.activation = activation
        self.leaky = leaky
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if self.leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        # instantiate layers
        if deform:
            # self.conv = AttDeformConv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.conv = DeformConv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(8, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)

class DWConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, leaky=True, norm='BATCH', activation=True):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        self.activation = activation
        self.leaky = leaky

        # activation, support PReLU and common ReLU
        if self.leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        # instantiate layers
        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, bias=False, groups=in_channels)
        self.pwconv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(8, in_channels)

    def basic_forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, lkdw=False, norm='BATCH', deform=False):
        super().__init__()
        self.norm_type = norm
        if leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)
        if lkdw:
            ## Large kernel 7x7
            # self.conv1 = DWConvNorm(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, leaky=leaky,
            #                         norm=norm, activation=True)
            # self.conv2 = DWConvNorm(out_channels, out_channels, kernel_size=7, stride=1, padding=3, leaky=leaky,
            #                         norm=norm, activation=True)
            self.conv1 = DWConvNorm(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, leaky=leaky,
                                    norm=norm, activation=True)
            self.conv2 = DWConvNorm(out_channels, out_channels, kernel_size=3, stride=1, padding=1, leaky=leaky,
                                    norm=norm, activation=True)
        else:
            self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True, deform)
            self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out

class ResFormerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, lkdw=False, norm='BATCH', deform=False):
        super().__init__()
        self.norm_type = norm
        if leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)
        if lkdw:
            ## Large Kernel  7 x 7
            # self.conv1 = DWConvNorm(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, leaky=leaky,
            #                         norm=norm, activation=True)
            # self.conv2 = DWConvNorm(out_channels, out_channels, kernel_size=7, stride=1, padding=3, leaky=leaky,
            #                         norm=norm, activation=True)
            ## +可变形卷积之后，采用3x3的DW
            self.conv1 = DWConvNorm(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, leaky=leaky,
                                    norm=norm, activation=True)
            self.conv2 = DWConvNorm(out_channels, out_channels, kernel_size=3, stride=1, padding=1, leaky=leaky,
                                    norm=norm, activation=True)
        else:
            self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True, deform=deform)
            self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        _out = self.conv1(x)
        out = self.conv2(_out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return _out, out


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        middle_channels = in_channels // 4
        if leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, middle_channels, 1, 1, leaky, norm, True)
        self.conv2 = ConvNorm(middle_channels, middle_channels, 3, stride, leaky, norm, True)
        self.conv3 = ConvNorm(middle_channels, out_channels, 1, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ScaleUpsample(nn.Module):
    def __init__(self, use_deconv=False, num_channels=None, scale_factor=None, mode='trilinear', align_corners=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        if use_deconv:
            self.trans_conv = nn.ConvTranspose3d(num_channels, num_channels, kernel_size=3,
                                                stride=scale_factor, padding=1, output_padding=scale_factor - 1)

    def forward(self, x):
        if not self.use_deconv:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        else:
            return self.trans_conv(x)


class AttentionConnection(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.param = nn.Parameter(torch.Tensor(1).fill_(factor))

    def forward(self, feature, attention):
        return (self.param + attention) * feature
