import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            # padding added to maintain spatial dimensions
            nn.Conv2d(3, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # padding added to maintain spatial dimensions
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # additional conv layer to reduce dimensions
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_input_size = self._get_fc_input_size(3)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _get_fc_input_size(self, num_channels):
        dummy_input = torch.randn(1, num_channels, 600, 800)
        dummy_output = self.localization(dummy_input)
        return int(np.prod(dummy_output.size()))

    def forward(self, x):
        # Transform input
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_input_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv1d(in_, out_, ks, st, pad),
        nn.BatchNorm1d(out_),
        nn.ReLU(inplace=True)
    )
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        res= self.act(self.bn(self.conv(x)))
        return res

    def fuseforward(self, x):
        res = self.act(self.conv(x))

        return res
    
class MAM2(nn.Module):
    def __init__(self, in_channel):
        super(MAM2, self).__init__()
        #self.T_projection = nn.Conv1d(257,197,kernel_size=1)
        self.channel264 = nn.Sequential(
            Conv(in_channel, in_channel//2, 3, 2, 1),
            convblock(in_channel//2, in_channel//4, 3, 1, 1),
            convblock(in_channel//4, in_channel//8, 3, 1, 0),
            convblock(in_channel//8, in_channel//16, 3, 1, 1),
            convblock(in_channel//16, 16, 1, 1, 1),
        )
        self.xy = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(16, 2, 1, 1, 0)
        )
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(16, 1, 1, 1, 0)
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(16, 1, 1, 1, 0)
        )
        # Start with identity transformation
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        self.scale1[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale1[-1].bias.data.zero_()
        self.scale2[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale2[-1].bias.data.zero_()
        # self.warp = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Conv1d(16, 2, 1, 1, 0)
        # )
        # self.warp[-1].weight.data.normal_(mean=0.0, std=5e-4)
        # self.warp[-1].bias.data.zero_()
        # self.fus1 = Conv(in_channel * 2, in_channel, 1, 1, 0)
    def forward(self,samples2,embeddings1,embeddings2):
        #embeddings1 = self.T_projection(embeddings1)
        # in_ = torch.cat([gr, gt], dim=1)
        in_ = embeddings1 - embeddings2
        in_ = in_.permute(0, 2, 1)  
        n1 = self.channel264(in_)
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
        # if in_.is_cuda:
        #     identity_theta = identity_theta.cuda().detach()
        shift_xy = self.xy(n1).view(-1, 2)
        shift_s1 = self.scale1(n1).view(-1)
        shift_s2 = self.scale2(n1).view(-1)
        # warp_amount = self.warp(n1).view(-1, 2)

        bsize = shift_xy.shape[0]
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1).cuda()
        identity_theta[:, :, 2] += shift_xy
        identity_theta[:, 0, 0] += shift_s1
        identity_theta[:, 1, 1] += shift_s2
        # identity_theta = identity_theta.half()
        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), samples2.size(), align_corners=True)
        # wrap_grid += warp_amount.view(bsize, 1, 1, 2)
        wrap_gr = F.grid_sample(samples2, wrap_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # fuse = self.fus1(torch.cat([wrap_gr,gt],dim=1))
        # map_rgb = torch.unsqueeze(torch.mean(wrap_gr, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(80, 80), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("31.png")
        return wrap_gr,shift_xy
    
if __name__ == '__main__':
    model = MAM2(1024).cuda()
    a = torch.randn(2, 257, 1024).cuda()
    b = torch.randn(2, 197, 1024).cuda()
    c = torch.randn(2, 3, 600, 800).cuda()
    m_warp = model(c,a,b)
    print(m_warp.shape)