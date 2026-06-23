import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 0) 一些通用组件：Conv2d / ResConv2d
# =========================================================
def _make_norm(norm, num_channels: int):
    if norm is None:
        return nn.Identity()
    # norm 可能是 nn.BatchNorm2d / nn.InstanceNorm2d 这样的类
    return norm(num_channels)

def _make_act(act):
    if act is None:
        return nn.Identity()
    if isinstance(act, str):
        a = act.lower()
        if a in ["relu"]:
            return nn.ReLU(inplace=True)
        if a in ["leakyrelu", "lrelu"]:
            return nn.LeakyReLU(0.1, inplace=True)
        if a in ["silu", "swish"]:
            return nn.SiLU(inplace=True)
        if a in ["gelu"]:
            return nn.GELU()
        if a in ["tanh"]:
            return nn.Tanh()
        raise ValueError(f"Unknown act: {act}")
    if isinstance(act, nn.Module):
        return act
    raise ValueError(f"act must be None/str/nn.Module, got {type(act)}")

class Conv2d(nn.Module):
    """
    兼容你原工程用法的封装：
    Conv2d(in_c, out_c, k, stride=1, padding=?, dilation=?, norm=?, act=?)
    """
    def __init__(
        self, in_c, out_c,
        kernel_size=3, stride=1, padding=1, dilation=1,
        bias=None, norm=nn.BatchNorm2d, act="leakyrelu",
        groups=1
    ):
        super().__init__()
        if bias is None:
            bias = (norm is None)

        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups
        )
        self.norm = _make_norm(norm, out_c)
        self.act = _make_act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResConv2d(nn.Module):
    """一个轻量 residual block：Conv -> Conv + skip"""
    def __init__(
        self, in_c, out_c,
        kernel_size=3, stride=1, padding=1, dilation=1,
        norm=nn.BatchNorm2d, act="leakyrelu"
    ):
        super().__init__()
        self.conv1 = Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, norm=norm, act=act)
        self.conv2 = Conv2d(out_c, out_c, kernel_size, stride=1, padding=padding, dilation=dilation, norm=norm, act=None)
        self.act = _make_act(act)

        if in_c != out_c or stride != 1:
            self.skip = Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, norm=norm, act=None)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        s = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + s)


# =========================================================
# 1) 自适应 SpatialTransformer：输入 disp 为“像素位移”
# =========================================================
class SpatialTransformer(nn.Module):
    """
    warp(src, disp_pix)：
      src: (B,C,H,W)
      disp_pix: (B,2,H,W) or (B,H,W,2) 像素位移 (dx,dy)
    内部自动转为 grid_sample 需要的 normalized disp，然后 grid + disp。
    """
    def __init__(self, mode="bilinear", padding_mode="zeros", align_corners=True, disp_in_pixels=True):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.disp_in_pixels = disp_in_pixels

        self.register_buffer("_grid", torch.empty(0), persistent=False)
        self._hw = None

    def _make_grid(self, H, W, device, dtype):
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # H,W
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)  # 1,H,W,2

    def _get_grid(self, H, W, device, dtype):
        if self._grid.numel() == 0 or self._hw != (H, W) or self._grid.device != device or self._grid.dtype != dtype:
            self._grid = self._make_grid(H, W, device, dtype)
            self._hw = (H, W)
        return self._grid

    def _pix2norm(self, disp, H, W):
        # disp: B,H,W,2 (pixels) -> normalized displacement
        if self.align_corners:
            sx = 2.0 / (W - 1) if W > 1 else 0.0
            sy = 2.0 / (H - 1) if H > 1 else 0.0
        else:
            sx = 2.0 / W if W > 0 else 0.0
            sy = 2.0 / H if H > 0 else 0.0
        disp = disp.clone()
        disp[..., 0] *= sx
        disp[..., 1] *= sy
        return disp

    def forward(self, src, disp):
        if disp.dim() == 4 and disp.shape[1] == 2:
            disp = disp.permute(0, 2, 3, 1)  # B,H,W,2
        B, H, W, _ = disp.shape
        grid = self._get_grid(H, W, disp.device, disp.dtype)  # 1,H,W,2

        if self.disp_in_pixels:
            disp = self._pix2norm(disp, H, W)

        flow = grid + disp
        return F.grid_sample(
            src, flow,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )


def upsample_flow(flow_pix, size_hw, align_corners=True):
    """
    光流/位移上采样（像素单位必须做幅度缩放）：
      flow_pix: (B,2,H,W) in pixels
      size_hw: (H_new, W_new)
    """
    H_old, W_old = flow_pix.shape[-2], flow_pix.shape[-1]
    H_new, W_new = int(size_hw[0]), int(size_hw[1])

    if H_old == H_new and W_old == W_new:
        return flow_pix

    flow_up = F.interpolate(flow_pix, size=(H_new, W_new), mode="bilinear", align_corners=align_corners)

    # 像素位移缩放（align_corners=True 时更严格用 (new-1)/(old-1)）
    if align_corners:
        sx = (W_new - 1) / (W_old - 1) if W_old > 1 else 1.0
        sy = (H_new - 1) / (H_old - 1) if H_old > 1 else 1.0
    else:
        sx = W_new / max(W_old, 1)
        sy = H_new / max(H_old, 1)

    flow_up[:, 0] *= sx  # dx
    flow_up[:, 1] *= sy  # dy
    return flow_up


# =========================================================
# 2) DispEstimator / DispRefiner：统一输出“像素位移”，并保留 unfold
# =========================================================
class DispEstimator(nn.Module):
    """
    预测像素位移 disp_pix: (B,2,H,W)
    保留 unfold 形成 local cost volume（ks*ks 通道），再用卷积头回归位移。
    """
    def __init__(
        self, channel, depth=4,
        norm=nn.BatchNorm2d,
        corrks=7, corr_dilation=4,
        max_disp_px=300.0,
        smooth_cost=True
    ):
        super().__init__()
        self.corrks = int(corrks)
        self.corr_dilation = int(corr_dilation)
        self.max_disp_px = float(max_disp_px)
        self.smooth_cost = bool(smooth_cost)

        # 预处理：两路拼接后一起过卷积（与你原版一致思路）
        self.preprocessor = Conv2d(channel, channel, 3, act=None, norm=None, dilation=1, padding=1)

        # 压缩（把 [feat1,feat2] 压到 channel）
        self.featcompressor = nn.Sequential(
            Conv2d(channel * 2, channel * 2, 3, padding=1, norm=norm, act="leakyrelu"),
            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None),
        )

        # 主干回归头：输入维度 = channel + ks^2
        ic = channel + (self.corrks ** 2)
        oc = channel
        dilation = 1
        layers = nn.ModuleList()
        for _ in range(depth - 1):
            oc = max(16, oc // 2)  # 防止太小
            layers.append(Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm, act="leakyrelu"))
            ic = oc
            dilation *= 2
        layers.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        self.layers = layers

    def localcorr(self, feat1, feat2):
        """
        feat1/feat2: (B,C,H,W)
        输出 corr: (B, channel + ks^2, H, W)
        """
        feat = self.featcompressor(torch.cat([feat1, feat2], dim=1))

        b, c, h, w = feat1.shape

        # 可选：用轻量平滑代替大核 Gaussian（更快）
        if self.smooth_cost:
            feat1_smooth = F.avg_pool2d(feat1, kernel_size=3, stride=1, padding=1)
        else:
            feat1_smooth = feat1

        # unfold 得到局部窗口块： (B, C*ks^2, H*W)
        pad = self.corr_dilation * (self.corrks - 1) // 2
        blk = F.unfold(
            feat1_smooth,
            kernel_size=self.corrks,
            dilation=self.corr_dilation,
            padding=pad,
            stride=1
        )  # B, C*ks^2, H*W

        # reshape -> (B,C,ks^2,H,W)
        blk = blk.view(b, c, self.corrks * self.corrks, h, w)

        # 代价：MSE（与你原版一致：feat2 - feat1_local）
        # local_cost: (B, ks^2, H, W)
        local_cost = (feat2.unsqueeze(2) - blk).pow(2).mean(dim=1)

        corr = torch.cat([feat, local_cost], dim=1)
        return corr

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.shape

        # 两路一起预处理（与你原逻辑一致）
        feat = torch.cat([feat1, feat2], dim=0)
        feat = self.preprocessor(feat)
        feat1p, feat2p = feat[:b], feat[b:]

        x = self.localcorr(feat1p, feat2p)
        for layer in self.layers:
            x = layer(x)

        # 输出像素位移：用 tanh 做软约束，比 clamp 更稳定
        disp_pix = self.max_disp_px * torch.tanh(x / self.max_disp_px)
        return disp_pix


class DispRefiner(nn.Module):
    """
    输入 disp_pix（像素单位），回归 delta_disp_pix，再相加得到 refined disp_pix。
    """
    def __init__(self, channel, dilation=1, depth=4, norm=nn.BatchNorm2d, max_delta_px=50.0):
        super().__init__()
        self.max_delta_px = float(max_delta_px)

        self.preprocessor = nn.Sequential(
            Conv2d(channel, channel, 3, dilation=dilation, padding=dilation, norm=None, act=None)
        )
        self.featcompressor = nn.Sequential(
            Conv2d(channel * 2, channel * 2, 3, padding=1, norm=norm, act="leakyrelu"),
            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None),
        )

        oc = channel
        ic = channel + 2
        dilation_ = 1
        estimator = nn.ModuleList()
        for _ in range(depth - 1):
            oc = max(16, oc // 2)
            estimator.append(Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation_, dilation=dilation_, norm=norm, act="leakyrelu"))
            ic = oc
            dilation_ *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        self.estimator = nn.Sequential(*estimator)

    def forward(self, feat1, feat2, disp_pix):
        b = feat1.shape[0]

        feat = torch.cat([feat1, feat2], dim=0)
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b], feat[b:]], dim=1))

        x = torch.cat([feat, disp_pix], dim=1)
        delta = self.estimator(x)
        delta = self.max_delta_px * torch.tanh(delta / self.max_delta_px)

        return disp_pix + delta


# =========================================================
# 3) Feature_extractor_unshare / FuseModule（按你原版保留）
# =========================================================
class Feature_extractor_unshare(nn.Module):
    def __init__(self, depth, base_ic, base_oc, base_dilation, norm):
        super().__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i % 2 == 1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(ResConv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            else:
                feature_extractor.append(Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            ic = oc
            if i % 2 == 1 and i < depth - 1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FuseModule(nn.Module):
    """Interactive fusion module（你原版保留）"""
    def __init__(self, in_dim=64):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, prior):
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)

        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out


# =========================================================
# 4) DenseMatcher：全程像素位移 + unfold 的 matcher + 自适应 warp
# =========================================================
class DenseMatcher(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, num_pyramids=2):
        super().__init__()
        self.num_pyramids = num_pyramids

        # unshare
        self.feature_extractor_unshare1 = Feature_extractor_unshare(
            depth=unshare_depth, base_ic=3, base_oc=8, base_dilation=1, norm=nn.InstanceNorm2d
        )
        self.feature_extractor_unshare2 = Feature_extractor_unshare(
            depth=unshare_depth, base_ic=3, base_oc=8, base_dilation=1, norm=nn.InstanceNorm2d
        )

        base_oc = self.feature_extractor_unshare1.oc

        # share pyramids: 1/2, 1/4, 1/8
        self.feature_extractor_share1 = nn.Sequential(
            Conv2d(base_oc, base_oc * 2, kernel_size=3, stride=1, padding=1, dilation=1, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 2, base_oc * 2, kernel_size=3, stride=2, padding=1, dilation=1, norm=nn.InstanceNorm2d),
        )
        self.feature_extractor_share2 = nn.Sequential(
            Conv2d(base_oc * 2, base_oc * 4, kernel_size=3, stride=1, padding=2, dilation=2, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 4, base_oc * 4, kernel_size=3, stride=2, padding=2, dilation=2, norm=nn.InstanceNorm2d),
        )
        self.feature_extractor_share3 = nn.Sequential(
            Conv2d(base_oc * 4, base_oc * 8, kernel_size=3, stride=1, padding=4, dilation=4, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 8, base_oc * 8, kernel_size=3, stride=2, padding=4, dilation=4, norm=nn.InstanceNorm2d),
        )

        # matcher：统一输出像素位移
        self.matcher_fine = DispEstimator(base_oc * 4, depth=matcher_depth, corrks=7, corr_dilation=4, max_disp_px=300.0, norm=nn.BatchNorm2d)
        self.matcher_coarse = DispEstimator(base_oc * 8, depth=matcher_depth, corrks=7, corr_dilation=4, max_disp_px=300.0, norm=nn.BatchNorm2d)

        # refiner：像素位移 refine
        self.refiner = DispRefiner(base_oc * 2, dilation=1, depth=4, norm=nn.BatchNorm2d, max_delta_px=50.0)

        # 自适应 warp（disp=像素）
        self.warp = SpatialTransformer(align_corners=True, disp_in_pixels=True)

    def match_one(self, feat11, feat12, feat21, feat22, feat31, feat32):
        """
        返回：disp_pix at feat11 分辨率（1/2）
        """
        # 1) coarse @ 1/8
        disp3_pix = self.matcher_coarse(feat31, feat32)  # (B,2,h8,w8)

        # 2) upsample to 1/4 + warp feat21
        disp3_to_2 = upsample_flow(disp3_pix, (feat21.shape[-2], feat21.shape[-1]), align_corners=True)
        feat21_w = self.warp(feat21, disp3_to_2)

        # 3) fine residual @ 1/4
        disp2_res = self.matcher_fine(feat21_w, feat22)  # pixels @ 1/4
        disp2_pix = disp3_to_2 + disp2_res

        # 4) upsample to 1/2 + warp feat11
        disp2_to_1 = upsample_flow(disp2_pix, (feat11.shape[-2], feat11.shape[-1]), align_corners=True)
        feat11_w = self.warp(feat11, disp2_to_1)

        # 5) refine @ 1/2
        disp1_pix = self.refiner(feat11_w, feat12, disp2_to_1)

        # 可选：轻量平滑（比大核 Gaussian 快很多）
        disp1_pix = F.avg_pool2d(disp1_pix, kernel_size=3, stride=1, padding=1)

        return disp1_pix, disp2_pix, disp3_pix  # 方便你训练时加多尺度监督

    def forward(self, src, tgt, type='ir2vis'):
        """
        src/tgt: (B,3,H,W)
        输出 dict，disp 全是“像素位移”，分辨率为 H,W
        """
        b, c, h, w = tgt.shape

        feat01 = self.feature_extractor_unshare1(src)
        feat02 = self.feature_extractor_unshare2(tgt)

        feat0 = torch.cat([feat01, feat02], dim=0)
        feat1 = self.feature_extractor_share1(feat0)  # 1/2
        feat2 = self.feature_extractor_share2(feat1)  # 1/4
        feat3 = self.feature_extractor_share3(feat2)  # 1/8

        feat11, feat12 = feat1[:b], feat1[b:]
        feat21, feat22 = feat2[:b], feat2[b:]
        feat31, feat32 = feat3[:b], feat3[b:]

        if type == 'bi':
            disp_12_1, disp_12_2, disp_12_3 = self.match_one(feat11, feat12, feat21, feat22, feat31, feat32)
            disp_21_1, disp_21_2, disp_21_3 = self.match_one(feat12, feat11, feat22, feat21, feat32, feat31)

            disp_12 = upsample_flow(disp_12_1, (h, w), align_corners=True)
            disp_21 = upsample_flow(disp_21_1, (h, w), align_corners=True)

            out = {'ir2vis': disp_12, 'vis2ir': disp_21}
            if self.training:
                # 多尺度输出（像素位移）：你想加 loss 时直接用
                out.update({
                    'ir2vis_down4': disp_12_2, 'ir2vis_down8': disp_12_3,
                    'vis2ir_down4': disp_21_2, 'vis2ir_down8': disp_21_3,
                })
            return out

        elif type == 'vis2ir':
            disp_21_1, disp_21_2, disp_21_3 = self.match_one(feat12, feat11, feat22, feat21, feat32, feat31)
            disp_21 = upsample_flow(disp_21_1, (h, w), align_corners=True)
            out = {'vis2ir': disp_21}
            if self.training:
                out.update({'vis2ir_down4': disp_21_2, 'vis2ir_down8': disp_21_3})
            return out

        else:  # 'ir2vis'
            disp_12_1, disp_12_2, disp_12_3 = self.match_one(feat11, feat12, feat21, feat22, feat31, feat32)
            disp_12 = upsample_flow(disp_12_1, (h, w), align_corners=True)
            out = {'ir2vis': disp_12}
            if self.training:
                out.update({'ir2vis_down4': disp_12_2, 'ir2vis_down8': disp_12_3})
            return out
