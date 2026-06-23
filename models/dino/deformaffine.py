import torch
import torch.nn as nn
# from models.layers import SpatialTransformer, ResizeTransform, conv_block, predict_flow, conv2D, MatchCost
import numpy as np

shape = (600, 800)
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

gpu_use = True


def construct_M(angle, scale_x, scale_y, center_x, center_y):
    alpha = torch.cos(angle)
    beta = torch.sin(angle)
    tx = center_x
    ty = center_y
    tmp0 = torch.cat((scale_x * alpha, beta), 1)
    tmp1 = torch.cat((-beta, scale_y * alpha), 1)
    theta = torch.cat((tmp0, tmp1), 0)
    t = torch.cat((tx, ty), 0)
    matrix = torch.cat((theta, t), 1)
    return theta, matrix


class ConstuctRotationLayer(nn.Module):
    def __init__(self):
        super(ConstuctRotationLayer, self).__init__()

    def forward(self, angle):
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        tmp0 = torch.cat((alpha, beta), 1)
        tmp1 = torch.cat((-beta, alpha), 1)
        theta = torch.cat((tmp0, tmp1), 0)
        t = torch.tensor([[0.], [0.]]).cuda()
        matrix = torch.cat((theta, t), 1)
        return theta, matrix


class ConstuctmatrixLayer(nn.Module):
    def __init__(self):
        super(ConstuctmatrixLayer, self).__init__()

    def forward(self, angle, scale_x, scale_y, center_x, center_y):
        theta, matrix = construct_M(angle, scale_x, scale_y, center_x, center_y)
        return theta, matrix


class AffineToFlow(nn.Module):

    def __init__(self, volsize):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(AffineToFlow, self).__init__()

        # Create sampling grid
        self.size = volsize

    def forward(self, matrix):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        flow = F.affine_grid(matrix.unsqueeze(0), [1, 1, self.size[0], self.size[1]], align_corners=True)
        shape = flow.shape[1:3]
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)

        for i in range(len(shape)):
            flow[:, i, ...] = (flow[:, i, ...].clone() / 2 + 0.5) * (shape[i] - 1)

        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        flow_offset = flow - grid

        return flow_offset


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border', align_corners=True), new_locs


class PointSpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(PointSpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, point, flow, intep=False):
        """
        Push the src and flow through the spatial transform block
            :param point: [N, 2]
            :param flow: the output from the U-Net [*vol_shape, 2]
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        new_point = point.clone().detach()

        if intep:
            for i in range(point.shape[1]):
                x_trunc, x_frac = new_point[0, i, 0].trunc(), new_point[0, i, 0].frac()
                y_trunc, y_frac = new_point[0, i, 1].trunc(), new_point[0, i, 1].frac()
                x0, y0 = x_trunc.long(), y_trunc.long()
                x1, y1 = (x_trunc+1).long(), y_trunc.long()
                x2, y2 = x_trunc.long(), (y_trunc+1).long()
                x3, y3 = (x_trunc+1).long(), (y_trunc+1).long()
                # dic ={'0': x_frac * y_frac, '1': (1-x_frac) * y_frac,
                #       '2': x_frac * (1-y_frac), '3': (1-x_frac) * (1-y_frac)}

                dic = {'0': x_frac * y_frac, '2': (1 - x_frac) * y_frac,
                       '1': x_frac * (1 - y_frac), '3': (1 - x_frac) * (1 - y_frac)}

                tmp_x = dic['0'] * new_locs[0, x0, y0, 0] + dic['1'] * new_locs[0, x1, y1, 0] +\
                               dic['2'] * new_locs[0, x2, y2, 0] + dic['3'] * new_locs[0, x3, y3, 0]
                tmp_y = dic['0'] * new_locs[0, x0, y0, 1] + dic['1'] * new_locs[0, x1, y1, 1] +\
                               dic['2'] * new_locs[0, x2, y2, 1] + dic['3'] * new_locs[0, x3, y3, 1]

                new_point[0, i, 1] = (tmp_x + 1) / 2 * 512
                new_point[0, i, 0] = (tmp_y + 1) / 2 * 512
        else:
            for i in range(point.shape[1]):
                x = min(new_point[0, i, 0].round().long(), 511)
                y = min(new_point[0, i, 1].round().long(), 511)
                new_point[0, i, 1] = (new_locs[0, x, y, 0] + 1) / 2 * 512
                new_point[0, i, 0] = (new_locs[0, x, y, 1] + 1) / 2 * 512

        return new_point


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out


def composition_flows(g1, g2):
    """
    warping an image twice, first with g1 then with g2
    :param g1, g2 is dense_flow/ offset
    :return:
    """
    transformer = SpatialTransformer(volsize=(512, 512))
    flow = g2 + transformer(g1, g2)
    return flow


def predict_flow(in_planes, d=3):
    dim = d
    conv_fn = getattr(nn, 'Conv%dd' % dim)
    return conv_fn(in_planes, dim, kernel_size=3, padding=1)


def conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def MatchCost(features_t, features_s):
    mc = torch.norm(features_t - features_s, p=1, dim=1) # torch.Size([1, 64, 64])
    mc = mc[..., np.newaxis] # np.newaxis: Extended dimension torch.Size([1, 64, 64, 1])
    return mc.permute(0, 3, 1, 2)


class DeformableNet2(nn.Module):
    def __init__(self):
        super(DeformableNet, self).__init__()
        # int_steps = 7   #
        self.inshape = shape

        down_shape2 = [int(d / 4) for d in self.inshape] # [64, 64]
        down_shape1 = [int(d / 2) for d in self.inshape] # [128, 128]
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform   = SpatialTransformer(volsize=self.inshape)

        # FeatureLearning/Encoder functions
        dim = 2
        self.enc = nn.ModuleList()


        # Dncoder functions
        od = 32 + 1
        self.conv2_0 = conv_block(dim, od, 48, 1) # [48, 32, 16]
        self.enc.append(self.conv2_0)
        self.conv2_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv2_1)
        self.conv2_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv2_2)
        self.predict_flow2a = predict_flow(16, 2)
        self.enc.append(self.predict_flow2a)

        self.dc_conv2_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1) # [48, 48, 32]
        self.enc.append(self.dc_conv2_0)
        self.dc_conv2_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv2_1)
        self.dc_conv2_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv2_2)
        self.predict_flow2b = predict_flow(32, 2)
        self.enc.append(self.predict_flow2b)

        self.resize = ResizeTransform(1 / 2, dim)

    def load_state_dict(self, state_dict, strict = False):
        state_dict.pop('spatial_transform.grid')
        state_dict.pop('spatial_transform_f.grid')
        super().load_state_dict(state_dict, strict)

    def forward(self, rgb, t):
        ##################### Estimation at scale-2 #######################
        corr2 = MatchCost(t, rgb)    # torch.Size([16,  1, 64, 64])红外可见光
        x = torch.cat((corr2, t), 1) # torch.Size([16, 33, 64, 64])
        x = self.conv2_0(x) # torch.Size([16, 48, 64, 64])
        x = self.conv2_1(x) # torch.Size([16, 32, 64, 64])
        x = self.conv2_2(x) # torch.Size([16, 16, 64, 64])
        flow2 = self.predict_flow2a(x) # torch.Size([16, 2, 64, 64]) flow2: flow field
        upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])

        x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])

        refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        int_flow2 = refine_flow2
        up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128])
        features_s_warped, _ = self.spatial_transform_f(t, up_int_flow2) # torch.Size([16, 16, 128, 128])

        return features_s_warped
    
class DeformableNet(nn.Module):
    def __init__(self):
        super(DeformableNet, self).__init__()
        # int_steps = 7   #
        self.inshape = shape

        down_shape2 = [int(d / 4) for d in self.inshape] # [64, 64]
        down_shape1 = [int(d / 2) for d in self.inshape] # [128, 128]
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform   = SpatialTransformer(volsize=self.inshape)

        # FeatureLearning/Encoder functions
        dim = 2
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 3, 16, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 16, 16, 1))  # 1
        self.enc.append(conv_block(dim, 16, 16, 1))  # 2
        self.enc.append(conv_block(dim, 16, 32, 2))  # 3
        self.enc.append(conv_block(dim, 32, 32, 1))  # 4
        self.enc.append(conv_block(dim, 32, 32, 1))  # 5


        # Dncoder functions
        od = 32 + 1
        self.conv2_0 = conv_block(dim, od, 48, 1) # [48, 32, 16]
        self.enc.append(self.conv2_0)
        self.conv2_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv2_1)
        self.conv2_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv2_2)
        self.predict_flow2a = predict_flow(16, 2)
        self.enc.append(self.predict_flow2a)

        self.dc_conv2_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1) # [48, 48, 32]
        self.enc.append(self.dc_conv2_0)
        self.dc_conv2_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv2_1)
        self.dc_conv2_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv2_2)
        self.predict_flow2b = predict_flow(32, 2)
        self.enc.append(self.predict_flow2b)

        od = 1 + 16 + 16 + 2
        self.conv1_0 = conv_block(dim, od, 48, 1)
        self.enc.append(self.conv1_0)
        self.conv1_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv1_1)
        self.conv1_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv1_2)
        self.predict_flow1a = predict_flow(16, 2)
        self.enc.append(self.predict_flow1a)

        self.dc_conv1_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.enc.append(self.dc_conv1_0)
        self.dc_conv1_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv1_1)
        self.dc_conv1_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv1_2)
        self.predict_flow1b = predict_flow(32, 2)
        self.enc.append(self.predict_flow1b)

        self.resize = ResizeTransform(1 / 2, dim)
        # self.integrate2 = VecInt(down_shape2, int_steps)
        # self.integrate1 = VecInt(down_shape1, int_steps)

    def load_state_dict(self, state_dict, strict = False):
        state_dict.pop('spatial_transform.grid')
        state_dict.pop('spatial_transform_f.grid')
        super().load_state_dict(state_dict, strict)

    def forward(self, tgt, src, shape=None):
        if shape is not None:
            down_shape1 = [int(d / 2) for d in shape]
            self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
            self.spatial_transform   = SpatialTransformer(volsize=shape)
        ##################### Feature extraction #########################
        c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 16, 128, 128])
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 16, 128, 128])
        c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 32, 64, 64])
        c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 32, 64, 64])

        ##################### Estimation at scale-2 #######################
        corr2 = MatchCost(c22, c12)    # torch.Size([16,  1, 64, 64])
        x = torch.cat((corr2, c22), 1) # torch.Size([16, 33, 64, 64])
        x = self.conv2_0(x) # torch.Size([16, 48, 64, 64])
        x = self.conv2_1(x) # torch.Size([16, 32, 64, 64])
        x = self.conv2_2(x) # torch.Size([16, 16, 64, 64])
        flow2 = self.predict_flow2a(x) # torch.Size([16, 2, 64, 64]) flow2: flow field
        upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])

        x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])

        refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        int_flow2 = refine_flow2
        # int_flow2 = self.integrate2(refine_flow2)
        up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128])
        features_s_warped, _ = self.spatial_transform_f(c11, up_int_flow2) # torch.Size([16, 16, 128, 128])


        ##################### Estimation at scale-1 #######################
        corr1 = MatchCost(c21, features_s_warped) # torch.Size([16, 1, 128, 128])
        x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1) # torch.Size([16, 35, 112, 112])
        x = self.conv1_0(x) # torch.Size([16, 48, 128, 128])
        x = self.conv1_1(x) # torch.Size([16, 32, 128, 128])
        x = self.conv1_2(x) # torch.Size([16, 16, 128, 128])
        flow1 = self.predict_flow1a(x) + up_int_flow2 # torch.Size([16, 2, 128, 128])

        x = self.dc_conv1_0(flow1) # torch.Size([16, 48, 128, 128])
        x = self.dc_conv1_1(x) # torch.Size([16, 48, 128, 128])
        x = self.dc_conv1_2(x) # torch.Size([16, 32, 128, 128])
        refine_flow1 = self.predict_flow1b(x) + flow1 # torch.Size([16, 2, 128, 128])
        int_flow1 = refine_flow1
        # int_flow1 = self.integrate1(refine_flow1)

        ##################### Upsample to scale-0 #######################
        flow = self.resize(int_flow1) # torch.Size([16, 2, 256, 256])
        m_warp, disp_pre = self.spatial_transform(src, flow) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])
        # wd+
        f_warp, _ = self.spatial_transform(tgt, (-flow)) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])

        return m_warp, f_warp, flow, int_flow1, int_flow2, disp_pre


def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()


if __name__ == '__main__':
    model = DeformableNet().cuda()
    a = torch.randn(2, 1, 256, 256).cuda()
    b = torch.randn(2, 1, 256, 256).cuda()
    m_warp = model(a,b)
    print(m_warp.shape)