'''
Diffeomorphic VoxelMorph

Original code retrieved from:
https://github.com/uncbiag/easyreg/tree/541eb8a4776b4d5c4a8cf15c109bf5aed22ebcd9

Original paper:
Dalca, A. V., Balakrishnan, G., Guttag, J., & Sabuncu, M. R. (2019).
Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces.
Medical image analysis, 57, 226-236.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''


import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.normal import Normal
from functools import partial
from timm.models.layers import DropPath, trunc_normal_, to_3tuple

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, D):
        B, _, C = x.shape
        
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
 
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W, D):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W, D))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 ca_num_heads=4, 
                 sa_num_heads=8,
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None):
        super().__init__()
        
        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.split_groups = self.dim//ca_num_heads
        
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_conv_1 = nn.Conv3d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2,1,1), padding=(1+i,0,0), stride=1, groups=dim//self.ca_num_heads)
            local_conv_2 = nn.Conv3d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(1,3+i*2,1), padding=(0,1+i,0), stride=1, groups=dim//self.ca_num_heads)
            local_conv_3 = nn.Conv3d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(1,1,3+i*2), padding=(0,0,1+i), stride=1, groups=dim//self.ca_num_heads)                
            setattr(self, f"local_conv_{i + 1}_1", local_conv_1)
            setattr(self, f"local_conv_{i + 1}_2", local_conv_2)
            setattr(self, f"local_conv_{i + 1}_3", local_conv_3)
        self.proj0 = nn.Conv3d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
        self.bn = nn.BatchNorm3d(dim*expand_ratio)
        self.proj1 = nn.Conv3d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        self.dw_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        
        v = self.v(x)
        s = self.s(x).reshape(B, H, W, D, self.ca_num_heads, C//self.ca_num_heads).permute(4,0,5,1,2,3)
        for i in range(self.ca_num_heads):
            local_conv_1 = getattr(self, f"local_conv_{i + 1}_1")
            local_conv_2 = getattr(self, f"local_conv_{i + 1}_2")
            local_conv_3 = getattr(self, f"local_conv_{i + 1}_3")
            s_i= s[i]
            s_i = local_conv_1(s_i)
            s_i = local_conv_2(s_i)
            s_i = local_conv_3(s_i).reshape(B, self.split_groups, -1, H, W, D)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out,s_i],2)


        s_out = s_out.reshape(B, C, H, W, D)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out)))).reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Block(nn.Module):

    def __init__(self, 
                 dim, 
                 ca_num_heads, 
                 sa_num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None):
        super().__init__()

        self.init_cfg = init_cfg

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention, 
            expand_ratio=expand_ratio,init_cfg=None)
            
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop,
            init_cfg=None)


    def forward(self, x, H, W, D):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, D))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        return x
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, D):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and D % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, D, C)

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    


    
class ScaBlock(nn.Module):
    def __init__(self, 
                 in_chans=32, 
                 num_classes=3, 
                 embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1], 
                 sa_num_heads=[-1, -1, 8, 16],
                 mlp_ratios=[4, 4, 4, 2], 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], 
                 ca_attentions=[1, 1, 1, 0],
                 num_stages=4,
                 head_conv=3,
                 expand_ratio=2,
                 convert_weights=True,
                 frozen_stages=-1):

        self.convert_weights = convert_weights

        super(ScaBlock, self).__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.frozen_stages = frozen_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        
        self.patch_embed = PatchEmbed(
            patch_size=2, in_chans=self.in_chans, embed_dim=embed_dims[0],
            norm_layer=norm_layer)

        for i in range(num_stages):
            block = nn.ModuleList([Block(
                dim=embed_dims[i], 
                ca_num_heads=ca_num_heads[i], 
                sa_num_heads=sa_num_heads[i],
                mlp_ratio=mlp_ratios[i], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                ca_attention=0 if i==2 and j%2!=0 else ca_attentions[i], 
                expand_ratio=expand_ratio,
                init_cfg=None)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            downsample=PatchMerging(dim=embed_dims[i], norm_layer=norm_layer, reduce_factor=4)
            cur += depths[i]
            
            setattr(self, f"downsample{i + 1}", downsample)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
    

    def forward(self, x):
        outs=[]
        x = self.patch_embed(x)
        B, _ ,H, W, D= x.size()
        x = x.flatten(2).transpose(1, 2)     
        for i in range(self.num_stages):
            downsample = getattr(self, f"downsample{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            for blk in block:
                x_out = blk(x, H, W, D)
            x_down = downsample(x_out, H, W, D) if (i < self.num_stages - 1) else x
            wH, wW, wD = (H + 1) // 2, (W + 1) // 2, (D + 1) // 2
            x_out = norm(x_out)
            out = x_out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
            outs.append(out)

            x, H, W, D = x_down, wH, wW, wD
        return tuple(outs)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

        self.act = nn.GELU()
        self.bn = nn.BatchNorm3d(planes)
    def forward(self, x):

        shortcut =  x
        out = self.conv1(x)

        out = self.act(self.bn(self.conv2(out)))


        out += shortcut
        out = out
        return out

class ScaMorph(nn.Module):
    def __init__(self, inshape):
        super(ScaMorph, self).__init__()
        self.inshape = inshape
        bias_opt = True

        self.registrationhead = self.head(2, 32, kernel_size=3, stride=1, bias=bias_opt)
        
        self.ec = ScaBlock(
            in_chans=32, 
            num_classes=16, 
            embed_dims=[32, 64, 128, 256], 
            ca_num_heads=[4, 4, 4, 4], 
            sa_num_heads=[-1, -1, 8, 16],
            mlp_ratios=[4, 4, 4, 2], 
            qkv_bias=True,
            drop_path_rate=0.2,
            depths=[2, 2, 4, 2],
            ca_attentions=[1, 1, 1, 1],
            num_stages=4,
            head_conv=3,
            expand_ratio=2)
        
        self.dc1 = self.encoder(256+128, 256, kernel_size=3, stride=1, bias=bias_opt)
        self.dc2 = self.encoder(256, 128, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(128 + 64, 128, kernel_size=3, stride=1, bias=bias_opt)
        self.dc4 = self.encoder(128, 64, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(64+32, 64, kernel_size=3, stride=1, bias=bias_opt)
        self.dc6 = self.encoder(64, 32, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.outputs(32, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.up1 = self.upsampling(256, 256)
        self.up2 = self.upsampling(128, 128)
        self.up3 = self.upsampling(64, 64)
        self.up4 = self.upsampling(32, 32)
        
        self.res0 = PreActBlock(32, 32, bias=bias_opt)
        self.res1 = PreActBlock(64, 64, bias=bias_opt)
        self.res2 = PreActBlock(128, 128, bias=bias_opt)
        self.res3 = PreActBlock(256, 256, bias=bias_opt)
        self.res4 = PreActBlock(256+128, 256+128, bias=bias_opt)
        self.res5 = PreActBlock(128+64, 128+64, bias=bias_opt)
        self.res6 = PreActBlock(64+32, 64+32, bias=bias_opt)

        ndims = len(self.inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        
        # configure transformer
        self.transformer = SpatialTransformer(self.inshape)
        
    def head(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU()            
            )
        return layer 
            
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU())
        return layer
        
    def upsampling(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer    
    
    def forward(self, x):
        source = x[:, 0:1, :, :]
        
        input = self.registrationhead(x)
        
        [e0,e1,e2,e3] = self.ec(input)
        
        d0 = torch.cat((self.up1(e3), e2), 1)

        d0 = self.dc1(self.res4(d0))
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e1), 1)

        d1 = self.dc3(self.res5(d1))
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e0), 1)

        d2 = self.dc5(self.res6(d2))
        d2 = self.dc6(d2)
        
        f_xy = self.dc7(self.up4(d2))

        flow_field = self.flow(f_xy)
        y_source = self.transformer(source, flow_field)
        
        return y_source, flow_field

