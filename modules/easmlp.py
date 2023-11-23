import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import sys
# from net.asmlp.shift_cuda import Shift, torch_shift, torch_shift_reverse
def torch_shift(x, shift_size, dim):
    B_, C, H, W = x.shape
    pad = shift_size // 2

    x = F.pad(x, (pad, pad, pad, pad) , "constant", 0)
    xs = torch.chunk(x, shift_size, 1)
    x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-pad, pad+1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, pad, H)
    x_cat = torch.narrow(x_cat, 3, pad, W)
    return x_cat

def torch_shift_reverse(x, shift_size, dim):
    B_, C, H, W = x.shape
    pad = shift_size // 2

    x = F.pad(x, (pad, pad, pad, pad) , "constant", 0)
    xs = torch.chunk(x, shift_size, 1)
    x_shift = [torch.roll(x_c, -shift, dim) for x_c, shift in zip(xs, range(-pad, pad+1))]  # 逆向移位
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, pad, H)
    x_cat = torch.narrow(x_cat, 3, pad, W)
    return x_cat

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AxialShift(nn.Module):
    r""" Axial shift

    Args:
        dim (int): Number of input channels.
        shift_size (int): shift size .
        as_bias (bool, optional):  If True, add a learnable bias to as mlp. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, shift_size, as_bias=True, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)

        self.actn = nn.GELU()

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)

        # cuda加速
        # self.shift_dim2 = Shift(self.shift_size, 2)
        # self.shift_dim3 = Shift(self.shift_size, 3)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, C, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)

        '''
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)

        xs = torch.chunk(x, self.shift_size, 1)

        def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

        x_shift_lr = shift(3)
        x_shift_td = shift(2)
        '''

        x_shift_lr = torch_shift(x, shift_size=self.shift_size, dim=3)  # W方向：左右平移，对齐水平方向相邻的两个像素点到同一位置
        # + @jy
        x_shift_l_diagon = torch_shift(x_shift_lr, shift_size=self.shift_size, dim=2)  # 先对W方向左右平移，再对H方向前后平移，对齐左对角线方向相邻的两个顶点像素到同一位置

        x_shift_td = torch_shift(x, shift_size=self.shift_size, dim=2)  # H方向：上下平移，对齐竖直方向相邻的两个像素点到同一位置
        # + @jy
        x_shift_r_diagon = torch_shift_reverse(x_shift_td, shift_size=self.shift_size, dim=3)  # 先对H方向前后平移，再对W方向右左（注意，这里反向）平移，对齐左对角线方向相邻的两个顶点像素到同一位置

        x_lr = self.actn(self.conv2_1(x_shift_lr))
        x_shift_l_diagon = self.actn(self.conv2_1(x_shift_l_diagon))
        x_td = self.actn(self.conv2_2(x_shift_td))
        x_shift_r_diagon = self.actn(self.conv2_2(x_shift_r_diagon))

        x = x_lr + x_shift_l_diagon + x_td + x_shift_r_diagon
        x = self.norm2(x)

        x = self.conv3(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, shift_size={self.shift_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # conv1
        flops += N * self.dim * self.dim
        # norm 1
        flops += N * self.dim
        # conv2_1 conv2_2
        flops += N * self.dim * self.dim * 2
        # x_lr + x_td
        flops += N * self.dim
        # norm2
        flops += N * self.dim
        # norm3
        flops += N * self.dim * self.dim
        return flops

class AxialShiftedBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        as_bias (bool, optional): If True, add a learnable bias to Axial Mlp. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size=7,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.axial_shift = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape

        shortcut = x
        x = self.norm1(x)

        # axial shift block
        x = self.axial_shift(x)  # B, C, H, W

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # shift mlp
        flops += self.axial_shift.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, shift_size,
                 mlp_ratio=4., as_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
                              shift_size=shift_size,
                              mlp_ratio=mlp_ratio,
                              as_bias=as_bias,
                              drop=drop,
                              drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                              norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)#.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

def MyNorm(dim):
    return nn.GroupNorm(1, dim)

class ASPMLP(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2],
                 shift_size=5, mlp_ratio=4., as_bias=True,
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=MyNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               shift_size=shift_size,
                               mlp_ratio=self.mlp_ratio,
                               as_bias=as_bias,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            # + @jy
            self.layers.append(layer)

        self.conv1x1 = nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=1, stride=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.conv1x1(x)  # (B, embed_dim, H, W) -> (B, in_chans*img_size, H, W)
        return x

# AsmlpBlockMerge2Channel
class AspmlpBlock(nn.Module):
    def __init__(self, img_size=16, patch_size=1, in_chans=512, out_chans=512, embed_dim=768, depth=18, shift_size=5):
        super(AspmlpBlock, self).__init__()
        self.aspmlp = ASPMLP(img_size=img_size, patch_size=patch_size, in_chans=in_chans*img_size, embed_dim=embed_dim,
                          depths=[depth], shift_size=shift_size)

        self.fuse_conv = nn.Conv3d(in_channels=in_chans * 3, out_channels=out_chans, kernel_size=1, stride=1)

    # 三维数据(H,W,D)，分别对三个二维面（hw,hd,wd）做MLP，再聚合
    ## 合并维度到通道C上
    def forward(self, x):
        B, C, H, W, D = x.shape
        # W-D维度
        df_wd = x.reshape(B, C * H, W, D)
        df_wd = self.aspmlp(df_wd).reshape(B, C, H, W, D)  # ->(B, C, H, W, D)
        # H-D维度
        df_hd = x.permute(0, 1, 3, 2, 4).reshape(B, C * W, H, D)
        # 等价于 df_hd = x.rearrange(B, (C, W), H, D) ?未测试过
        df_hd = self.aspmlp(df_hd).reshape(B, C, W, H, D).permute(0, 1, 3, 2, 4)  # ->(B, C, H, W, D)
        # H-W维度
        df_hw = x.permute(0, 1, 4, 2, 3).reshape(B, C * D, H, W)
        df_hw = self.aspmlp(df_hw).reshape(B, C, D, H, W).permute(0, 1, 3, 4, 2)  # ->(B, C, H, W, D)

        # 三个轴向特征融合：concat + 1x1 Conv
        x = self.fuse_conv(torch.cat([df_wd, df_hd, df_hw], dim=1))   # (B, 3*C, H, W, D) ->(B, C, H, W, D)

        return x

# AsmlpBlockMerge2Channel  | AspmlpBlock_Light
class AspmlpBlock_Light(nn.Module):
    def __init__(self, img_size=16, patch_size=1, in_chans=512, out_chans=512, embed_dim=768, depth=18, shift_size=5):
        super(AspmlpBlock_Light, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_channels=in_chans*img_size, out_channels=in_chans, kernel_size=1, stride=1)
        self.aspmlp = ASPMLP(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                             depths=[depth], shift_size=shift_size)
        self.conv1x1_up = nn.Conv2d(in_channels=in_chans, out_channels=in_chans*img_size, kernel_size=1, stride=1)

        self.fuse_conv = nn.Conv3d(in_channels=in_chans * 3, out_channels=out_chans, kernel_size=1, stride=1)

    # 三维数据(H,W,D)，分别对三个二维面（hw,hd,wd）做MLP，再聚合
    ## 合并维度到通道C上
    def forward(self, x):
        B, C, H, W, D = x.shape
        ## 对通道进行缩减，再进行MLP操作，然后再将通道数扩张恢复
        # W-D维度
        df_wd = self.conv1x1_down(x.reshape(B, C * H, W, D))
        df_wd = self.conv1x1_up(self.aspmlp(df_wd)).reshape(B, C, H, W, D)  # ->(B, C, H, W, D)
        # H-D维度
        df_hd = self.conv1x1_down(x.permute(0, 1, 3, 2, 4).reshape(B, C * W, H, D))
        # 等价于 df_hd = x.rearrange(B, (C, W), H, D) ?未测试过
        df_hd = self.conv1x1_up(self.aspmlp(df_hd)).reshape(B, C, W, H, D).permute(0, 1, 3, 2, 4)  # ->(B, C, H, W, D)
        # H-W维度
        df_hw = self.conv1x1_down(x.permute(0, 1, 4, 2, 3).reshape(B, C * D, H, W))
        df_hw = self.conv1x1_up(self.aspmlp(df_hw)).reshape(B, C, D, H, W).permute(0, 1, 3, 4, 2)  # ->(B, C, H, W, D)

        # 三个轴向特征融合：concat + 1x1 Conv
        x = self.fuse_conv(torch.cat([df_wd, df_hd, df_hw], dim=1))   # (B, 3*C, H, W, D) ->(B, C, H, W, D)

        return x

# AspmlpBlockMerge2Batch
class AspmlpBlock_Light_B(nn.Module):
    def __init__(self, img_size=16, patch_size=1, in_chans=512, out_chans=512, embed_dim=768, depth=18, shift_size=5):
        super(AspmlpBlock_Light_B, self).__init__()
        self.aspmlp = ASPMLP(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                          depths=[depth], shift_size=shift_size)

        self.fuse_conv = nn.Conv3d(in_channels=in_chans * 3, out_channels=out_chans, kernel_size=1, stride=1)

    # 三维数据(H,W,D)，分别对三个二维面（hw,hd,wd）做MLP，再聚合
    def forward(self, x):
        B, C, H, W, D = x.shape

        ## 合并维度到batch size(B)上
        # W-D维度
        df_wd = x.permute(0, 2, 1, 3, 4).reshape(B * H, C, W, D)  # -> (B*D, C, H, W)
        df_wd = self.aspmlp(df_wd).reshape(B, H, C, W, D).permute(0, 2, 1, 3, 4)  # ->(B, C, H, W, D)
        # H-D维度
        df_hd = x.permute(0, 3, 1, 2, 4).reshape(B * W, C, H, D)  # -> (B*W, C, H, D)
        df_hd = self.aspmlp(df_hd).reshape(B, W, C, H, D).permute(0, 2, 1, 3, 4)  # ->(B, C, H, W, D)
        # H-W维度
        df_hw = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
        df_hw = self.aspmlp(df_hw).reshape(B, D, C, H, W).permute(0, 2, 3, 4, 1)  # ->(B, C, H, W, D)

        # 三个轴向特征融合：concat + 1x1 Conv
        x = self.fuse_conv(torch.cat([df_wd, df_hd, df_hw], dim=1))   # (B, 3*C, H, W, D) ->(B, C, H, W, D)

        return x

class AspmlpBlock2(nn.Module):
    def __init__(self, img_size=16, patch_size=1, in_chans=512, out_chans=512, embed_dim=768, depth=18, shift_size=5):
        super(AspmlpBlock2, self).__init__()
        self.asmlp = ASPMLP(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                          depths=[depth], shift_size=shift_size)

        self.fuse_conv = nn.Conv3d(in_channels=in_chans * 3, out_channels=out_chans, kernel_size=1, stride=1)

    # 三维数据(H,W,D)，分别对三个二维面（hw,hd,wd）做MLP，再聚合
    def forward(self, x):
        B, C, H, W, D = x.shape

        ## 合并维度到batch size(B)上
        # W-D维度
        df_wd = x.reshape(B, C, H, W*D)  # -> (B, C, H, W*D)
        df_wd = self.asmlp(df_wd).reshape(B, C, H, W, D)  # ->(B, C, H, W, D)
        # H-D维度
        df_hd = x.permute(0, 1, 3, 2, 4).reshape(B, C, W, H*D)  # -> (B, C, H, W*D)
        df_hd = self.asmlp(df_hd).reshape(B, C, W, H, D).permute(0, 1, 3, 2, 4)  # ->(B, C, H, W, D)
        # H-W维度
        df_hw = x.permute(0, 1, 4, 2, 3).reshape(B, C, D, H*W)  # -> (B, C, D, H*W)
        df_hw = self.asmlp(df_hw).reshape(B, C, D, H, W).permute(0, 1, 3, 4, 2)  # ->(B, C, H, W, D)

        # ## 合并维度到通道C上
        # # W-D维度
        # df_wd = x.reshape(B, C * H, W, D)
        # df_wd = self.asmlp(df_wd).reshape(B, C, H, W, D)  # ->(B, C, H, W, D)
        # # H-D维度
        # df_hd = x.permute(0, 1, 3, 2, 4).reshape(B, C * W, H, D)
        # # 等价于 df_hd = x.rearrange(B, (C, W), H, D) ?未测试过
        # df_hd = self.asmlp(df_hd).reshape(B, C, W, H, D).permute(0, 1, 3, 2, 4)  # ->(B, C, H, W, D)
        # # H-W维度
        # df_hw = x.permute(0, 1, 4, 2, 3).reshape(B, C * D, H, W)
        # df_hw = self.asmlp(df_hw).reshape(B, C, D, H, W).permute(0, 1, 3, 4, 2)  # ->(B, C, H, W, D)

        # 三个轴向特征融合：concat + 1x1 Conv
        x = self.fuse_conv(torch.cat([df_wd, df_hd, df_hw], dim=1))   # (B, 3*C, H, W, D) ->(B, C, H, W, D)

        return x

if __name__ == '__main__':
    # model = Bottleneck_ACM(inplanes=128, planes=128)
    model = ASPMLP(img_size=20, patch_size=1, in_chans=512, embed_dim=512,
                  depths=[2, 2, 3, 2], shift_size=3)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'There are {n_params} trainable parameters.')

    x = torch.randn([1, 512, 20, 20])

    y = model(x)
    print('y.shape:', y.shape)