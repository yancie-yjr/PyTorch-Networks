import torch
import torch.nn as nn
import torchvision
import numpy as np

class SpatialPyramidPooling(nn.Module):
    def __init__(self, output_sizes = [1, 4, 8, 12]):
        super(SpatialPyramidPooling, self).__init__()

        self.pool_layers = nn.ModuleList()
        for output_size in output_sizes:
            self.pool_layers.append(nn.AdaptiveMaxPool2d(output_size=output_size))

    def forward(self, x):
        outputs = []
        b, c, w, h = x.size()
        for pool_layer in self.pool_layers:
            outputs.append(pool_layer(x).view(b, c, -1))
            # print(pool_layer(x).size())
        out = torch.cat(outputs, dim=2)
        # print('out',out.size())
        return out

class APNB(nn.Module):
    def __init__(self, channel):
        super(APNB, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class AFNB(nn.Module):
    def __init__(self, channel):
        super(AFNB, self).__init__()
        self.inter_channel = channel // 2
        self.output_sizes = [1, 3, 6, 8]
        self.sample_dim = np.sum([size*size for size in self.output_sizes])
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta_spp = SpatialPyramidPooling(self.output_sizes)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g_spp = SpatialPyramidPooling(self.output_sizes)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        # [N, C, H , W]
        b, c, h, w = x.size()
        b_y, c_y, h_y, w_y = y.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, C/2]
        # yyy = self.conv_theta_spp(self.conv_theta(y))
        # print(yyy.shape)


        y_theta = self.conv_theta_spp(self.conv_theta(y))
        y_g = self.conv_g_spp(self.conv_g()).permute(0, 2, 1).contiguous()

        # print('x',x.size())
        # print('y', y.size())
        # print('x_phi', x_phi.size())
        # print('y_theta', y_theta.size())
        # print('self.conv_g_spp(self.conv_g(x))',self.conv_g_spp(self.conv_g(x)).size())
        # print('x_g',x_g.size())

        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_phi, y_theta)
        mul_theta_phi = self.softmax(mul_theta_phi)

        # print('mul_theta_phi',mul_theta_phi.size())
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, y_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out