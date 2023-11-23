# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PoolFormer implementation
"""
import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

from models.attention_blocks import CBAM

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 4, 'input_size': (4, 128, 128), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=1, stride=1, padding=0,
                 in_chans=4, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ConvBlock(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, in_chans=64, out_chans=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)
        # self.conv = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

class SppBlock(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, in_chans=64, out_chans=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=7, stride=1, padding=3)

        self.proj = nn.Conv2d(4 * out_chans, out_chans, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.proj(out)

class AsppBlock(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, in_chans=64, out_chans=64):
        super().__init__()
        self.aspp = ASPP(in_chans, (6, 12, 18), out_chans)

    def forward(self, x):
        return self.aspp(x)

class ConvAttentionBlock(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, in_chans=64, out_chans=64):
        super().__init__()
        self.att = CBAM(in_chans)
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        return self.conv(self.att(x))


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # self.token_mixer = Pooling(pool_size=pool_size)  # poolFormer
        self.token_mixer = ConvAttentionBlock(dim, dim)  # ConvBlock | ConvAttentionBlock | AsppBlock | SppBlock
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, in_chans=4, embed_dims=None,
                 mlp_ratios=None, downsamples=None, 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None, 
                 pretrained=None, 
                 **kwargs):

        super().__init__()

        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=in_chans, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])

        self.apply(self.cls_init_weights)

        self.conv1x1 = nn.Conv2d(in_channels=embed_dims[-1], out_channels=in_chans, kernel_size=1, stride=1)

        self.init_cfg = copy.deepcopy(init_cfg)
    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        x = self.conv1x1(x)
        return x
        # cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        # return cls_out

# AsmlpBlockMerge2Channel
class PoolFormerBlock3D(nn.Module):
    def __init__(self, img_size=16, layers=[6], patch_size=1, in_chans=128, out_chans=128, embed_dim=768):
        super(PoolFormerBlock3D, self).__init__()
        self.poolformer = PoolFormer(layers=layers, in_chans=in_chans*img_size, embed_dims=[embed_dim], mlp_ratios=[4],
                                    downsamples=[False], pool_size=patch_size, in_patch_size=1, in_stride=1, in_pad=0)

        self.fuse_conv = nn.Conv3d(in_channels=in_chans * 3, out_channels=out_chans, kernel_size=1, stride=1)

    # 三维数据(H,W,D)，分别对三个二维面（hw,hd,wd）做MLP，再聚合
    def forward(self, x):
        B, C, H, W, D = x.shape

        # ## 合并维度到通道C上
        # W-D维度
        df_wd = x.reshape(B, C * H, W, D)
        df_wd = self.poolformer(df_wd).reshape(B, C, H, W, D)  # ->(B, C, H, W, D)
        # H-D维度
        df_hd = x.permute(0, 1, 3, 2, 4).reshape(B, C * W, H, D)
        # 等价于 df_hd = x.rearrange(B, (C, W), H, D) ?未测试过
        df_hd = self.poolformer(df_hd).reshape(B, C, W, H, D).permute(0, 1, 3, 2, 4)  # ->(B, C, H, W, D)
        # H-W维度
        df_hw = x.permute(0, 1, 4, 2, 3).reshape(B, C * D, H, W)
        df_hw = self.poolformer(df_hw).reshape(B, C, D, H, W).permute(0, 1, 3, 4, 2)  # ->(B, C, H, W, D)

        # 三个轴向特征融合：concat + 1x1 Conv
        x = self.fuse_conv(torch.cat([df_wd, df_hd, df_hw], dim=1))   # (B, 3*C, H, W, D) ->(B, C, H, W, D)

        return x

# AsmlpBlockMerge2Channel
class PoolFormerBlock3D_Light(nn.Module):
    def __init__(self, img_size=16, layers=[6], patch_size=1, in_chans=128, out_chans=128, embed_dim=768):
        super(PoolFormerBlock3D_Light, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_channels=in_chans * img_size, out_channels=in_chans, kernel_size=1, stride=1)
        self.poolformer = PoolFormer(layers=layers, in_chans=in_chans, embed_dims=[embed_dim],
                                     mlp_ratios=[4], downsamples=[False], pool_size=patch_size, in_patch_size=1,
                                     in_stride=1, in_pad=0)
        self.conv1x1_up = nn.Conv2d(in_channels=in_chans, out_channels=in_chans * img_size, kernel_size=1, stride=1)

        self.fuse_conv = nn.Conv3d(in_channels=in_chans * 3, out_channels=out_chans, kernel_size=1, stride=1)

    def forward(self, x):
        B, C, H, W, D = x.shape

        # ## merger to Channel
        # W-D
        df_wd = self.conv1x1_down(x.reshape(B, C * H, W, D))
        df_wd = self.conv1x1_up(self.poolformer(df_wd)).reshape(B, C, H, W, D)  # ->(B, C, H, W, D)
        # H-D
        df_hd = self.conv1x1_down(x.permute(0, 1, 3, 2, 4).reshape(B, C * W, H, D))
        # equal to df_hd = x.rearrange(B, (C, W), H, D)
        df_hd = self.conv1x1_up(self.poolformer(df_hd)).reshape(B, C, W, H, D).permute(0, 1, 3, 2, 4)  # ->(B, C, H, W, D)
        # H-W
        df_hw = self.conv1x1_down(x.permute(0, 1, 4, 2, 3).reshape(B, C * D, H, W))
        df_hw = self.conv1x1_up(self.poolformer(df_hw)).reshape(B, C, D, H, W).permute(0, 1, 3, 4, 2)  # ->(B, C, H, W, D)

        # concat + 1x1 Conv
        x = self.fuse_conv(torch.cat([df_wd, df_hd, df_hw], dim=1))   # (B, 3*C, H, W, D) ->(B, C, H, W, D)

        return x

# AsmlpBlockMerge2Batch
class PoolFormerBlock3D_B(nn.Module):
    def __init__(self, img_size=16, layers=[6], patch_size=1, in_chans=128, out_chans=128, embed_dim=768):
        super(PoolFormerBlock3D_B, self).__init__()
        self.poolformer = PoolFormer(layers=layers, in_chans=in_chans, embed_dims=[embed_dim],
                                     mlp_ratios=[4], downsamples=[False], pool_size=patch_size, in_patch_size=1,
                                     in_stride=1, in_pad=0)

        self.fuse_conv = nn.Conv3d(in_channels=in_chans * 3, out_channels=out_chans, kernel_size=1, stride=1)

    def forward(self, x):
        B, C, H, W, D = x.shape

        ## merger to batch
        # W-D
        df_wd = x.permute(0, 2, 1, 3, 4).reshape(B * H, C, W, D)  # -> (B*D, C, H, W)
        df_wd = self.poolformer(df_wd).reshape(B, H, C, W, D).permute(0, 2, 1, 3, 4)  # ->(B, C, H, W, D)
        # H-D
        df_hd = x.permute(0, 3, 1, 2, 4).reshape(B * W, C, H, D)  # -> (B*W, C, H, D)
        df_hd = self.poolformer(df_hd).reshape(B, W, C, H, D).permute(0, 2, 1, 3, 4)  # ->(B, C, H, W, D)
        # H-W
        df_hw = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
        df_hw = self.poolformer(df_hw).reshape(B, D, C, H, W).permute(0, 2, 3, 4, 1)  # ->(B, C, H, W, D)

        # concat + 1x1 Conv
        x = self.fuse_conv(torch.cat([df_wd, df_hd, df_hw], dim=1))   # (B, 3*C, H, W, D) ->(B, C, H, W, D)

        return x

model_urls = {
    "poolformer_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",
    "poolformer_s24": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",
    "poolformer_s36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",
    "poolformer_m36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",
    "poolformer_m48": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",
}

@register_model
def poolformer_backbone(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios:
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [4]
    embed_dims = [512]
    mlp_ratios = [4]
    downsamples = [False]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    return model

@register_model
def poolformer_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_s24(pretrained=False, **kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s24']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_s36(pretrained=False, **kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_m36(pretrained=False, **kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_m48(pretrained=False, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m48']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

# @jy
if __name__ == "__main__":
    # model = UNet_L4(num_classes=4, input_channels=4, channels=(16, 32, 64, 128),
    #              use_deconv=False, strides=(1, 2, 2, 2), leaky=True, norm='INSTANCE')

    model = PoolFormer(in_chans=4, embed_dims=[512], layers=[4], mlp_ratios=[4], downsamples=[False], pool_size=1,
                       num_classes=4, in_patch_size=1, in_stride=1, in_pad=0)

    # model = UNet_L4_GateCASPMLP(num_classes=4, input_channels=4, channels=(16, 32, 64, 128),
    #                        use_deconv=False, strides=(1, 2, 2, 2), leaky=True, norm='INSTANCE')

    # print("model: ", model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'There are {n_params} trainable parameters.')

    batch_size = 1
    x = torch.randn([batch_size, 4, 128, 128])

    # print FLOPs、Params
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
    print('y: ', y.shape)