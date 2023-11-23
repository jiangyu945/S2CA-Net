from models.basic_modules import *
from models.attention_blocks import FCNHead, ParallelDecoder, AttentionGate

from modules.easmlp import AspmlpBlock, AspmlpBlock_Light, AspmlpBlock_Light_B
from modules.attention_blocks import SENet3D, SpacialAttention3D

from models.poolformer import PoolFormerBlock3D, PoolFormerBlock3D_Light
from modules.DeformableBlock3D import AttDeformConv3d

head_list = ['fcn', 'parallel']
head_map = {'fcn': FCNHead,
            'parallel': ParallelDecoder}

class Backbone_L4(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=4, channels=(16, 32, 64, 128), strides=(1, 2, 2, 2), lkdw=False, deform=False, **kwargs):
        super().__init__()
        self.nb_filter = channels
        self.strides = strides
        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, self.nb_filter[0], kernel_size=3, stride=self.strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = res_unit(input_channels, self.nb_filter[0], self.strides[0], lkdw=lkdw, **kwargs)

        self.conv1_0 = res_unit(self.nb_filter[0], self.nb_filter[1], self.strides[1], lkdw=lkdw, **kwargs)
        self.conv2_0 = res_unit(self.nb_filter[1], self.nb_filter[2], self.strides[2], lkdw=lkdw, **kwargs)
        self.conv3_0 = res_unit(self.nb_filter[2], self.nb_filter[3], self.strides[3], lkdw=lkdw, deform=deform, **kwargs)

    def forward(self, x):
        x0_0 = self.conv0_0(x)     # (128, 128, 128) * 4  -> (128, 128, 128) * 16
        x1_0 = self.conv1_0(x0_0)  # (128, 128, 128) * 16 -> (64, 64, 64) * 32
        x2_0 = self.conv2_0(x1_0)  # (64, 64, 64) * 32 -> (32, 32, 32) * 64
        x3_0 = self.conv3_0(x2_0)  # (32, 32, 32) * 64 -> (16, 16, 16) * 128
        return x0_0, x1_0, x2_0, x3_0

class Backbone_L4_former(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=4, channels=(16, 32, 64, 128), strides=(1, 2, 2, 2), embed_dim=[360, 512],
                 layers=[6, 2], light=False, lkdw=False, deform=False, **kwargs):
        super().__init__()

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, channels[0], kernel_size=3, stride=strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = ResFormerBlock(input_channels, channels[0], strides[0], lkdw=lkdw, **kwargs)

        self.conv1_0 = ResFormerBlock(channels[0], channels[1], strides[1], deform=deform, **kwargs)  # lkdw=lkdw -> deform=deform
        self.conv2_0 = ResFormerBlock(channels[1], channels[2], strides[2], lkdw=lkdw, **kwargs) # , lkdw=lkdw
        self.conv3_0 = ResFormerBlock(channels[2], channels[3], strides[3], lkdw=lkdw, **kwargs) # , lkdw=lkdw
        if light:
            self.former2_0 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2],
                                               out_chans=channels[2], embed_dim=embed_dim[0])
            self.former3_0 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                               out_chans=channels[3], embed_dim=embed_dim[1])
        else:
            self.former2_0 = PoolFormerBlock3D(layers=[layers[0]], img_size=32, in_chans=channels[2],
                                             out_chans=channels[2], embed_dim=embed_dim[0])
            self.former3_0 = PoolFormerBlock3D(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                               out_chans=channels[3], embed_dim=embed_dim[1])

    def forward(self, x):
        _, x0_0 = self.conv0_0(x)  # (128, 128, 128) * 16
        _, x1_0 = self.conv1_0(x0_0)  # (64, 64, 64) * 32
        x2_, x2_0 = self.conv2_0(x1_0)  # (32, 32, 32) * 64
        x2_0 = x2_0 + self.former2_0(x2_)  # (32, 32, 32) * 64  CNN + Former
        x3_, x3_0 = self.conv3_0(x2_0)  # (16, 16, 16) * 128
        x3_0 = x3_0 + self.former3_0(x3_)  # (16, 16, 16) * 128
        return x0_0, x1_0, x2_0, x3_0

class Backbone_L4_LGSM(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=4, channels=(16, 32, 64, 128), strides=(1, 2, 2, 2), embed_dim=[360, 512],
                 layers=[6, 2], light=False, **kwargs):
        super().__init__()

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, channels[0], kernel_size=3, stride=strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = ResFormerBlock(input_channels, channels[0], strides[0], lkdw=True, **kwargs)

        self.conv1_0 = ResFormerBlock(channels[0], channels[1], strides[1], lkdw=True, **kwargs)  # lkdw=lkdw -> deform=deform
        self.conv2_0 = ResFormerBlock(channels[1], channels[2], strides[2], lkdw=True, **kwargs) # , lkdw=lkdw
        self.conv3_0 = ResFormerBlock(channels[2], channels[3], strides[3], lkdw=True, **kwargs) # , lkdw=lkdw
        if light:
            self.former2_0 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2],
                                               out_chans=channels[2], embed_dim=embed_dim[0])
            self.former3_0 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                               out_chans=channels[3], embed_dim=embed_dim[1])
        else:
            self.former2_0 = PoolFormerBlock3D(layers=[layers[0]], img_size=32, in_chans=channels[2],
                                             out_chans=channels[2], embed_dim=embed_dim[0])
            self.former3_0 = PoolFormerBlock3D(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                               out_chans=channels[3], embed_dim=embed_dim[1])

    def forward(self, x):
        _, x0_0 = self.conv0_0(x)  # (128, 128, 128) * 16
        _, x1_0 = self.conv1_0(x0_0)  # (64, 64, 64) * 32
        x2_, x2_0 = self.conv2_0(x1_0)  # (32, 32, 32) * 64
        x2_0 = x2_0 + self.former2_0(x2_)  # (32, 32, 32) * 64  CNN + Former
        x3_, x3_0 = self.conv3_0(x2_0)  # (16, 16, 16) * 128
        x3_0 = x3_0 + self.former3_0(x3_)  # (16, 16, 16) * 128
        return x0_0, x1_0, x2_0, x3_0

class Backbone_L4_former_L3(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=4, channels=(16, 32, 64, 128), strides=(1, 2, 2, 2), embed_dim=360,
                 layers=6, light=False, **kwargs):
        super().__init__()

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, channels[0], kernel_size=3, stride=strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = ResFormerBlock(input_channels, channels[0], strides[0], **kwargs)

        self.conv1_0 = ResFormerBlock(channels[0], channels[1], strides[1], **kwargs)
        self.conv2_0 = ResFormerBlock(channels[1], channels[2], strides[2], **kwargs)
        self.conv3_0 = ResFormerBlock(channels[2], channels[3], strides[3], **kwargs)
        if light:
            self.former2_0 = PoolFormerBlock3D_Light(layers=[layers], img_size=32, in_chans=channels[2],
                                               out_chans=channels[2], embed_dim=embed_dim)
        else:
            self.former2_0 = PoolFormerBlock3D(layers=[layers], img_size=32, in_chans=channels[2],
                                             out_chans=channels[2], embed_dim=embed_dim)

    def forward(self, x):
        _, x0_0 = self.conv0_0(x)  # (128, 128, 128) * 16
        _, x1_0 = self.conv1_0(x0_0)  # (64, 64, 64) * 32
        x2_, x2_0 = self.conv2_0(x1_0)  # (32, 32, 32) * 64
        x2_0 = x2_0 + self.former2_0(x2_)  # (32, 32, 32) * 64  CNN + Former
        x3_, x3_0 = self.conv3_0(x2_0)  # (16, 16, 16) * 128
        return x0_0, x1_0, x2_0, x3_0

class Backbone(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=3, channels=(32, 64, 128, 256, 512), strides=(1, 2, 2, 2, 2), **kwargs):
        super().__init__()
        self.nb_filter = channels
        # self.strides = strides + (5 - len(strides)) * (1,)  # strides -> (1, 2, 2, 2, 2)
        self.strides = strides
        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, self.nb_filter[0], kernel_size=3, stride=self.strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = res_unit(input_channels, self.nb_filter[0], self.strides[0], **kwargs)
        self.conv1_0 = res_unit(self.nb_filter[0], self.nb_filter[1], self.strides[1], **kwargs)
        self.conv2_0 = res_unit(self.nb_filter[1], self.nb_filter[2], self.strides[2], **kwargs)
        self.conv3_0 = res_unit(self.nb_filter[2], self.nb_filter[3], self.strides[3], **kwargs)
        self.conv4_0 = res_unit(self.nb_filter[3], self.nb_filter[4], self.strides[4], **kwargs)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        return x0_0, x1_0, x2_0, x3_0, x4_0


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        raise NotImplementedError('Forward method must be implemented before calling it!')

    def predictor(self, x):
        return self.forward(x)['out']

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        if is_ds:
            loss_3 = criterion(outputs['level3'], targets)
            loss_2 = criterion(outputs['level2'], targets)
            loss_1 = criterion(outputs['level1'], targets)
            multi_loss = loss_out + loss_3 + loss_2 + loss_1
        else:
            multi_loss = loss_out
        return multi_loss

class UNet_L4(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_2 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_3 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x2_1 = self.conv2_1(torch.cat([x2, self.up3_2(x3)], dim=1))  # chans： nb_filter[2] + nb_filter[3]
        x1_2 = self.conv1_2(torch.cat([x1, self.up2_1(x2_1)], dim=1))  # chans： nb_filter[1] + nb_filter[2]
        x0_3 = self.conv0_3(torch.cat([x0, self.up1_0(x1_2)], dim=1))  # chans： nb_filter[0] + nb_filter[1]

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_3), size=size, mode='trilinear', align_corners=False)
        # return out
        return out, x0_3  # for feature map output

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

# Unet_l4(3D Deformable Conv)
class UNet_L4_Deform(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides, deform=True, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)  # deform=True
        self.conv1_2 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_3 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x2_1 = self.conv2_1(torch.cat([x2, self.up3_2(x3)], dim=1))  # chans： nb_filter[2] + nb_filter[3]
        x1_2 = self.conv1_2(torch.cat([x1, self.up2_1(x2_1)], dim=1))  # chans： nb_filter[1] + nb_filter[2]
        x0_3 = self.conv0_3(torch.cat([x0, self.up1_0(x1_2)], dim=1))  # chans： nb_filter[0] + nb_filter[1]

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_3), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

class UNet_L4_LKDW(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides, lkdw=True, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResFormerBlock if channels[-1] <= 320 else ResBottleneck
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], lkdw=True, **kwargs)
        self.conv1_2 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], lkdw=True, **kwargs)
        self.conv0_3 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], lkdw=True, **kwargs)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x2_1 = self.conv2_1(torch.cat([x2, self.up3_2(x3)], dim=1))  # chans： nb_filter[2] + nb_filter[3]
        x1_2 = self.conv1_2(torch.cat([x1, self.up2_1(x2_1)], dim=1))  # chans： nb_filter[1] + nb_filter[2]
        x0_3 = self.conv0_3(torch.cat([x0, self.up1_0(x1_2)], dim=1))  # chans： nb_filter[0] + nb_filter[1]

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_3), size=size, mode='trilinear', align_corners=False)
        return out, x0_3

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

class UNet_L4_PoolFormer(SegmentationNetwork):
    def __init__(self, num_classes=4, input_channels=4, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides, light=True,
                                           **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[2], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=512)
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[6], img_size=32, in_chans=channels[2] + channels[3],
                                           out_chans=channels[2], embed_dim=360)
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        # deep supervision
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x3_d = self.conv3_1(x3) + self.former3_1(x3)
        x2_d_ = torch.cat([x2, self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)   # chans： nb_filter[2] + nb_filter[3] -> nb_filter[2]
        x1_d = self.conv1_1(torch.cat([x1, self.up2_1(x2_d)], dim=1))  # chans： nb_filter[1] + nb_filter[2] -> nb_filter[1]
        x0_d = self.conv0_1(torch.cat([x0, self.up1_0(x1_d)], dim=1))  # chans： nb_filter[0] + nb_filter[1] -> nb_filter[0]

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

class UNet_L4_ConvFormer(SegmentationNetwork):
    def __init__(self, num_classes=4, input_channels=4, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[360, 512], layers=[6, 2], **kwargs):
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides, light=True,
                                           **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                           out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        # deep supervision
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x3_d = self.conv3_1(x3) + self.former3_1(x3)  # CNN + Former
        x2_d_ = torch.cat([x2, self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)   # CNN + Former
        x1_d = self.conv1_1(torch.cat([x1, self.up2_1(x2_d)], dim=1))  # chans： nb_filter[1] + nb_filter[2] -> nb_filter[1]
        x0_d = self.conv0_1(torch.cat([x0, self.up1_0(x1_d)], dim=1))  # chans： nb_filter[0] + nb_filter[1] -> nb_filter[0]

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

## using CBAM+Conv1x1 as Tokenizer
class UNet_L4_ConvAttFormer(SegmentationNetwork):
    def __init__(self, num_classes=4, input_channels=4, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[360, 512], layers=[6, 2], **kwargs):
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides, light=True,
                                           embed_dim=embed_dim, layers=layers, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                           out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        # deep supervision
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x3_d = self.conv3_1(x3) + self.former3_1(x3)  # CNN + Former
        x2_d_ = torch.cat([x2, self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)   # CNN + Former
        x1_d = self.conv1_1(torch.cat([x1, self.up2_1(x2_d)], dim=1))  # chans： nb_filter[1] + nb_filter[2] -> nb_filter[1]
        x0_d = self.conv0_1(torch.cat([x0, self.up1_0(x1_d)], dim=1))  # chans： nb_filter[0] + nb_filter[1] -> nb_filter[0]

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        # return out
        return out, x0_d  ## for feature map output

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

class UNet_L4_ASPMLP(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_2 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_3 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.BottomAspmlp_l3 = AspmlpBlock(img_size=32, patch_size=1, in_chans=channels[0] * (1 + 2 + 4),
                                           out_chans=channels[0] * 4, embed_dim=768, depth=4, shift_size=5)
        self.ca1 = SENet3D(in_channel=channels[0] * (1 + 2 + 4))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=768, depth=4, shift_size=5)
        self.ca2 = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L3
        x02 = self.down4Pooling(x0)
        x12 = self.down2Pooling(x1)
        x2_ = self.ca1(torch.cat([x02, x12, x2], dim=1))
        x2_skip = self.BottomAspmlp_l3(x2_)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca2(torch.cat([x03, x13, x23, x3], dim=1))
        x3_skip = self.BottomAspmlp_l4(x3_)

        x2_d = self.conv2_1(torch.cat([x2_skip, self.up3_2(x3_skip)], dim=1))
        x1_d = self.conv1_2(torch.cat([x1, self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_3(torch.cat([x0, self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

class UNet_L4_GateCASPMLP(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_2 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_3 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        ## 多尺度，最大池化方式
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.BottomAspmlp_l3 = AspmlpBlock(img_size=32, patch_size=1, in_chans=channels[0] * (1 + 2 + 4),
                                           out_chans=channels[0] * 4, embed_dim=768, depth=4, shift_size=5)
        self.ca1 = SENet3D(in_channel=channels[0] * (1 + 2 + 4))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=768, depth=4, shift_size=5)
        self.ca2 = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gate2_1 = AttentionGate(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gate1_0 = AttentionGate(nb_filter[1], nb_filter[0], nb_filter[0])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L3
        x02 = self.down4Pooling(x0)
        x12 = self.down2Pooling(x1)
        x2_ = self.ca1(torch.cat([x02, x12, x2], dim=1))
        x2_skip = x2 + self.BottomAspmlp_l3(x2_)  # short-cut connection: CNN + MLP

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca2(torch.cat([x03, x13, x23, x3], dim=1))
        x3_skip = x3 + self.BottomAspmlp_l4(x3_)  # short-cut connection: CNN + MLP

        x2_d = self.conv2_1(torch.cat([self.gate3_2(x3_skip, x2_skip), self.up3_2(x3_skip)], dim=1))
        x1_d = self.conv1_2(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_3(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

class UNet_L4_Gate_CASPMLP(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_2 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_3 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        ## 多尺度，最大池化方式
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )


        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gate2_1 = AttentionGate(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gate1_0 = AttentionGate(nb_filter[1], nb_filter[0], nb_filter[0])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_skip = x3 + self.conv_bottom(x3) + self.BottomAspmlp_l4(x3_)  # short-cut connection: CNN + MLP

        x2_d = self.conv2_1(torch.cat([self.gate3_2(x3_skip, x2), self.up3_2(x3_skip)], dim=1))
        x1_d = self.conv1_2(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_3(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

## MLP merge 3rd spacial dimension to Batch
class UNet_L4_Gate_CASPMLP_B(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_2 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_3 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )


        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gate2_1 = AttentionGate(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gate1_0 = AttentionGate(nb_filter[1], nb_filter[0], nb_filter[0])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_skip = x3 + self.conv_bottom(x3) + self.BottomAspmlp_l4(x3_)  # short-cut connection: CNN + MLP

        x2_d = self.conv2_1(torch.cat([self.gate3_2(x3_skip, x2), self.up3_2(x3_skip)], dim=1))
        x1_d = self.conv1_2(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_3(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class UNet_L4_Former_CSMLP(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[360, 512], layers=[6, 2], **kwargs):  # [256, 512]  [6, 2]
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            ## fixed weights
            loss_weight = [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'], targets)

        multi_loss = loss_out
        return multi_loss

class UNet_L4_Former_CSMLP_Deform(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[360, 512], layers=[6, 2], **kwargs):  # [256, 512]  [6, 2]
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        ## using maxpooling
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'], targets)

        multi_loss = loss_out
        return multi_loss

class UNet_L4_Former_CSMLP_LKDW(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[360, 512], layers=[6, 2], lkdw=True, **kwargs):  # [256, 512]  [6, 2]
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, lkdw=lkdw, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=lkdw, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=lkdw, **kwargs)

        ## 多尺度，最大池化方式
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        # out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'], targets)

        multi_loss = loss_out
        return multi_loss

class UNet_L4_Former_CSMLP_LKDW_Deform(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[256, 512], layers=[4, 2], lkdw=True, **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, lkdw=lkdw, deform=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=lkdw, **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=lkdw, **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=lkdw, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=lkdw, **kwargs)

        ## 多尺度，最大池化方式
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], lkdw=lkdw, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=lkdw, **kwargs),
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out, x0_d  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss


class UNet_L4_ConvFormer_CSMLP(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[360, 512], layers=[6, 2], **kwargs):
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides,
                                           light=True, embed_dim=embed_dim, layers=layers, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # short-cut + Sigmoid(CNN) * MLP

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        # out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0
        else:
            loss_out = criterion(outputs['out'], targets)

        multi_loss = loss_out
        return multi_loss

class UNet_L4_Former_CSRMLP(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides,
                                           light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[2], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=512)
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[6], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=360)
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        ## 多尺度，最大池化方式
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)


        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # short-cut connection: Sigmoid(MLP) * CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss

class UNet_L4_Former_CSHMLP(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128),
                 use_deconv=False, strides=(1, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone_L4_former(input_channels=input_channels, channels=channels, strides=strides,
                                           light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[2], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=512)
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[6], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=360)
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], **kwargs)

        ## 多尺度，最大池化方式
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention
        self.ca = SENet3D(in_channel=channels[0] + channels[1] + channels[2] + channels[3])  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] + channels[1] + channels[2] + channels[3],
                                       out_chans=channels[3], embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] + channels[1] + channels[2] + channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs),
                                         res_unit(channels[3], channels[3], **kwargs)
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)


        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_) + \
                    self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # short-cut connection: Sigmoid(CNN) * MLP

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class UNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(1, 2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.up4_3(x4)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.up3_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.up2_1(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.up1_0(x1_3)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out, x0_4

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class AttentionUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(1, 2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides  # (1, 2, 2, 2, 2)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

        self.gate4_3 = AttentionGate(nb_filter[4], nb_filter[3], nb_filter[3])
        self.gate3_2 = AttentionGate(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gate2_1 = AttentionGate(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gate1_0 = AttentionGate(nb_filter[1], nb_filter[0], nb_filter[0])

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([self.gate4_3(x4, x3), self.up4_3(x4)], 1))
        x2_2 = self.conv2_2(torch.cat([self.gate3_2(x3_1, x2), self.up3_2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([self.gate2_1(x2_2, x1), self.up2_1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([self.gate1_0(x1_3, x0), self.up1_0(x1_3)], 1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out, x0_4

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class CascadedUNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.first_stage = UNet(1, input_channels, channels, use_deconv, strides, **kwargs)
        self.second_stage = UNet(num_classes, input_channels, channels, use_deconv, strides, **kwargs)

    def forward(self, x):
        roi = self.first_stage(x)['out']
        roi_ = (torch.sigmoid(roi) > 0.5).float()
        roi_input = x * (1 + roi_)
        fine_seg = self.second_stage(roi_input)['out']
        output = dict()
        output['stage1'] = roi
        output['out'] = fine_seg
        return output

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class EnhancedUNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.up4_3(x4)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.up3_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.up2_1(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.up1_0(x1_3)], dim=1))

        out = dict()
        out['level3'] = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        out['level1'] = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out


class PriorAttentionNet(SegmentationNetwork):
    """
    The proposed Prior Attention Network for 3D BraTS segmentation.
    """
    def __init__(self, num_classes, head='fcn', input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(1, 2, 2, 2, 2), **kwargs):
        super().__init__()
        assert head in head_list
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter

        self.head = head
        self.one_stage = head_map[head](in_channels=nb_filter[2:], out_channels=1)

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # downsample attention
        self.conv_down = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # parameterized skip connection
        self.skip_3 = AttentionConnection()
        self.skip_2 = AttentionConnection()
        self.skip_1 = AttentionConnection()
        self.skip_0 = AttentionConnection()

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        attention = self.one_stage(x2, x3, x4)

        act_attention = torch.sigmoid(attention)  # attention shape is the same as x2

        attention_x3 = self.conv_down(act_attention)
        attention3_1 = F.interpolate(attention_x3, size=x3.shape[2:], mode='trilinear', align_corners=False)
        attention2_2 = F.interpolate(act_attention, size=x2.shape[2:], mode='trilinear', align_corners=False)
        attention1_3 = F.interpolate(act_attention, size=x1.shape[2:], mode='trilinear', align_corners=False)
        attention0_4 = F.interpolate(act_attention, size=x0.shape[2:], mode='trilinear', align_corners=False)

        x3_1 = self.conv3_1(torch.cat([self.skip_3(x3, attention3_1), self.up4_3(x4)], dim=1))  # (nb_filter[3], H3, W3, D3)
        x2_2 = self.conv2_2(torch.cat([self.skip_2(x2, attention2_2), self.up3_2(x3_1)], dim=1))  # (nb_filter[2], H2, W2, D2)
        x1_3 = self.conv1_3(torch.cat([self.skip_1(x1, attention1_3), self.up2_1(x2_2)], dim=1))  # (nb_filter[1], H1, W1, D3)
        x0_4 = self.conv0_4(torch.cat([self.skip_0(x0, attention0_4), self.up1_0(x1_3)], dim=1))  # (nb_filter[0], H0, W0, D0)

        out = dict()
        out['stage1'] = F.interpolate(attention, size=size, mode='trilinear', align_corners=False)  # intermediate
        out['level3'] = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        out['level1'] = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out, x0_4

#==========================================================+ADC=======================================================#
class UNet_LGSM_MCA_Deform(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[256, 512], layers=[4, 2], **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4_LGSM(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        ## mutil-scale，stride conv
        # self.down8Pooling = nn.Conv3d(channels[0], channels[0], kernel_size=8, stride=8, padding=0)
        # self.down4Pooling = nn.Conv3d(channels[1], channels[1], kernel_size=4, stride=4, padding=0)
        # self.down2Pooling = nn.Conv3d(channels[2], channels[2], kernel_size=2, stride=2, padding=0)
        ## mutil-scale，maxpooling
        self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        self.skip_0 = AttDeformConv3d(channels[0], channels[0], se_ratio=4)
        self.skip_1 = AttDeformConv3d(channels[1], channels[1], se_ratio=4)
        self.skip_2 = AttDeformConv3d(channels[2], channels[2], se_ratio=4)

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, self.skip_2(x2)), self.up3_2(x3_d)], dim=1)  #
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, self.skip_1(x1)), self.up2_1(x2_d)], dim=1))  # self.skip_1(x1)
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))  # self.skip_0(x0)

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

class UNet_LGSM_MCA_MSADC_Plus(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[256, 512], layers=[4, 2], **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4_LGSM(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        ## mutil-scale，stride conv
        self.down8Pooling = nn.Conv3d(channels[0], channels[0], kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.Conv3d(channels[1], channels[1], kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.Conv3d(channels[2], channels[2], kernel_size=2, stride=2, padding=0)
        ## mutil-scale，maxpooling
        # self.down8Pooling = nn.MaxPool3d(kernel_size=8, stride=8, padding=0)
        # self.down4Pooling = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
        # self.down2Pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        self.in_down01 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        self.in_up21 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2, padding=0)
        self.skip_ca = SENet3D(in_channel=3*channels[1])

        self.skip_msadc = AttDeformConv3d(3*channels[1], 3*channels[1], se_ratio=3*4)

        self.out_up10 = nn.ConvTranspose3d(3 * channels[1], channels[0], kernel_size=2, stride=2, padding=0)
        self.out_1 = nn.Conv3d(3 * channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.out_down12 = nn.Conv3d(3 * channels[1], channels[2], kernel_size=2, stride=2, padding=0)

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x_skip_in = self.skip_ca(torch.cat([self.in_down01(x0), x1, self.in_up21(x2)], dim=1))
        x_skip_out = self.skip_msadc(x_skip_in)

        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP

        # Decoding
        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former channel[3]
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2 + self.out_down12(x_skip_out)), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former  channel[2]
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1 + self.out_1(x_skip_out)), self.up2_1(x2_d)], dim=1))  # channel[1]
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0 + self.out_up10(x_skip_out)), self.up1_0(x1_d)], dim=1))  # channel[0]

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            ## 固定权重
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

class UNet_LGSM_MCA_MSADC(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[360, 512], layers=[6, 2], **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4_LGSM(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        self.down8Pooling = nn.Conv3d(channels[0], channels[0], kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.Conv3d(channels[1], channels[1], kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.Conv3d(channels[2], channels[2], kernel_size=2, stride=2, padding=0)

        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        self.in_down01 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        self.in_up21 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2, padding=0)
        self.skip_ca = SENet3D(in_channel=3*channels[1])

        self.skip_msadc = AttDeformConv3d(3*channels[1], 3*channels[1], se_ratio=3*4)

        self.out_up10 = nn.ConvTranspose3d(3 * channels[1], channels[0], kernel_size=2, stride=2, padding=0)
        self.out_1 = nn.Conv3d(3 * channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.out_down12 = nn.Conv3d(3 * channels[1], channels[2], kernel_size=2, stride=2, padding=0)

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x_skip_in = self.skip_ca(torch.cat([self.in_down01(x0), x1, self.in_up21(x2)], dim=1))
        x_skip_out = self.skip_msadc(x_skip_in)
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, self.out_down12(x_skip_out)), self.up3_2(x3_d)], dim=1)  #
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, self.out_1(x_skip_out)), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, self.out_up10(x_skip_out)), self.up1_0(x1_d)], dim=1))  # self.skip_0(x0)

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out, x0_d  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

## Baseline + LGSM + MCA
class UNet_LGSM_MCA(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[256, 512], layers=[4, 2], **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4_LGSM(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        self.down8Pooling = nn.Conv3d(channels[0], channels[0], kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.Conv3d(channels[1], channels[1], kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.Conv3d(channels[2], channels[2], kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)  # x2 | self.skip_2(x2)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))  # x1 | self.1(x1)
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))  # self.skip_0(x0)

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

## Baseline + MCA
class UNet_MCA(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides,
                                           lkdw=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        self.down8Pooling = nn.Conv3d(channels[0], channels[0], kernel_size=8, stride=8, padding=0)
        self.down4Pooling = nn.Conv3d(channels[1], channels[1], kernel_size=4, stride=4, padding=0)
        self.down2Pooling = nn.Conv3d(channels[2], channels[2], kernel_size=2, stride=2, padding=0)
        self.ca = SENet3D(in_channel=channels[0] * (1 + 2 + 4 + 8))  # channel attention

        self.BottomAspmlp_l4 = AspmlpBlock_Light_B(img_size=16, patch_size=1, in_chans=channels[0] * (1 + 2 + 4 + 8),
                                       out_chans=channels[0] * 8, embed_dim=128, depth=4, shift_size=5)
        self.conv_bottom = nn.Sequential(res_unit(channels[0] * (1 + 2 + 4 + 8), channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         )
        self.sa = SpacialAttention3D(kernel_size=1)

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        # multi_scale fusion skip connection for L4
        x03 = self.down8Pooling(x0)
        x13 = self.down4Pooling(x1)
        x23 = self.down2Pooling(x2)
        x3_ = self.ca(torch.cat([x03, x13, x23, x3], dim=1))
        x3_bottom = x3 + self.sa(self.conv_bottom(x3_)) * self.BottomAspmlp_l4(x3_)  # CNN->MLP
        # x3_bottom = x3 + self.sa(self.BottomAspmlp_l4(x3_)) * self.conv_bottom(x3_)  # MLP->CNN

        x3_d = self.conv3_1(x3_bottom)
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)  # x2 | self.skip_2(x2)
        x2_d = self.conv2_1(x2_d_)
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))  # x1 | self.1(x1)
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))  # self.skip_0(x0)

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

## Baseline + LGSM
class UNet_LGSM(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[256, 512], layers=[4, 2], **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4_LGSM(input_channels=input_channels, channels=channels, strides=strides,
                                           embed_dim=embed_dim, light=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.former3_1 = PoolFormerBlock3D_Light(layers=[layers[1]], img_size=16, in_chans=channels[3],
                                                 out_chans=channels[3], embed_dim=embed_dim[1])
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.former2_1 = PoolFormerBlock3D_Light(layers=[layers[0]], img_size=32, in_chans=channels[2] + channels[3],
                                                 out_chans=channels[2], embed_dim=embed_dim[0])
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        self.conv_bottom = nn.Sequential(res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs)
                                         )


        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x3_bottom = self.conv_bottom(x3)
        x3_d = self.conv3_1(x3_bottom) + self.former3_1(x3_bottom)  # CNN + Former
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)
        x2_d = self.conv2_1(x2_d_) + self.former2_1(x2_d_)  # CNN + Former
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

## Baseline + MSADC
class UNet_MSADC(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides,
                                           lkdw=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        self.conv_bottom = nn.Sequential(res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs)
                                         )

        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        self.in_down01 = nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        self.in_up21 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2, padding=0)
        self.skip_ca = SENet3D(in_channel=3*channels[1])

        self.skip_msadc = AttDeformConv3d(3*channels[1], 3*channels[1], se_ratio=3*4)

        self.out_up10 = nn.ConvTranspose3d(3 * channels[1], channels[0], kernel_size=2, stride=2, padding=0)
        self.out_1 = nn.Conv3d(3 * channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.out_down12 = nn.Conv3d(3 * channels[1], channels[2], kernel_size=2, stride=2, padding=0)

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)

        x_skip_in = self.skip_ca(torch.cat([self.in_down01(x0), x1, self.in_up21(x2)], dim=1))
        x_skip_out = self.skip_msadc(x_skip_in)
        x3_bottom = self.conv_bottom(x3)

        x3_d = self.conv3_1(x3_bottom)
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2 + self.out_down12(x_skip_out)), self.up3_2(x3_d)], dim=1)  # self.skip_2(x2)
        x2_d = self.conv2_1(x2_d_)
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1 + self.out_1(x_skip_out)), self.up2_1(x2_d)], dim=1))
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0 + self.out_up10(x_skip_out)), self.up1_0(x1_d)], dim=1))  # self.skip_0(x0)

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

## Baseline
class Baseline(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(16, 32, 64, 128), use_deconv=False,
                 strides=(1, 2, 2, 2), embed_dim=[256, 512], layers=[4, 2], **kwargs):  # [360, 512]  [6, 2]  | [256,512] [4,2]
        super().__init__()
        self.backbone = Backbone_L4(input_channels=input_channels, channels=channels, strides=strides,
                                           lkdw=True, **kwargs)

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(channels[3], channels[3], lkdw=True, **kwargs)
        self.conv2_1 = res_unit(channels[2] + channels[3], channels[2], lkdw=True, **kwargs)
        self.conv1_1 = res_unit(channels[1] + channels[2], channels[1], lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] + channels[1], channels[0], lkdw=True, **kwargs)

        self.conv_bottom = nn.Sequential(res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs),
                                         res_unit(channels[3], channels[3], lkdw=True, **kwargs)
                                         )


        # upsample for the decoder
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])

        self.gate3_2 = AttentionGate(channels[3], channels[2], channels[2])
        self.gate2_1 = AttentionGate(channels[2], channels[1], channels[1])
        self.gate1_0 = AttentionGate(channels[1], channels[0], channels[0])

        # deep supervision
        # self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.convds3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3 = self.backbone(x)


        x3_d = self.conv3_1(self.conv_bottom(x3))
        x2_d_ = torch.cat([self.gate3_2(x3_d, x2), self.up3_2(x3_d)], dim=1)  # x2 | self.skip_2(x2)
        x2_d = self.conv2_1(x2_d_)
        x1_d = self.conv1_1(torch.cat([self.gate2_1(x2_d, x1), self.up2_1(x2_d)], dim=1))  # x1 | self.1(x1)
        x0_d = self.conv0_1(torch.cat([self.gate1_0(x1_d, x0), self.up1_0(x1_d)], dim=1))  # self.skip_0(x0)

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out  # , x0_d

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0

        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss

# @jy
if __name__ == "__main__":
    # UNet_L4 | UNet_L4_Gate_CASPMLP
    # UNet_L4_PoolFormer | UNet_L4_ConvAttFormer | UNet_L4_ConvFormer |
    # Net_L4_Former_CSMLP | UNet_L4_Former_CSHMLP
    # UNet_L4_Former_CSHMLP_DS | UNet_L4_ConvFormer_CSMLP
    # model = AttentionUNet(num_classes=4, input_channels=4, leaky=True, norm='INSTANCE')
    model = UNet_LGSM_MCA_MSADC_Plus(num_classes=4, input_channels=4, channels=(16, 32, 64, 128),   # (16, 32, 64, 128) | (8, 16, 32, 64) | (32, 64, 128, 256)
                    use_deconv=False, strides=(1, 2, 2, 2), leaky=True, norm='INSTANCE')

    # model = UNet_L4_GateCASPMLP(num_classes=4, input_channels=4, channels=(16, 32, 64, 128),
    #                        use_deconv=False, strsides=(1, 2, 2, 2), leaky=True, norm='INSTANCE')

    # print("model: ", model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'There are {n_params} trainable parameters.')

    batch_size = 1
    x = torch.randn([batch_size, 4, 128, 128, 128])

    # Calculate FLOPs and Params
    from thop import profile
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # GPU
    # model.train()
    # model.cuda()
    # y = model(x.cuda())

    # CPU
    y = model(x)
    print('y: ', y['out'].shape)