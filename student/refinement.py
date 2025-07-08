import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
# import matplotlib.pyplot as plt

from feature import BasicConv, Conv2x


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


# class StereoNetRefinement(nn.Module):
#     def __init__(self):
#         super(StereoNetRefinement, self).__init__()
#
#         # Original StereoNet: left, disp
#         self.conv = conv2d(4, 32)
#
#         self.dilation_list = [1, 2, 4, 8, 1, 1]
#         self.dilated_blocks = nn.ModuleList()
#
#         for dilation in self.dilation_list:
#             self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))
#
#         self.dilated_blocks = nn.Sequential(*self.dilated_blocks)
#
#         self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)
#
#     def forward(self, low_disp, left_img, right_img=None):
#         """Upsample low resolution disparity prediction to
#         corresponding resolution as image size
#         Args:
#             low_disp: [B, H, W]
#             left_img: [B, 3, H, W]
#             right_img: [B, 3, H, W]
#         """
#         assert low_disp.dim() == 4
#         # low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
#         scale_factor = left_img.size(-1) / low_disp.size(-1)
#         disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
#         disp = disp * scale_factor  # scale correspondingly
#
#         concat = torch.cat((disp, left_img), dim=1)  # [B, 4, H, W]
#         out = self.conv(concat)
#         out = self.dilated_blocks(out)
#         residual_disp = self.final_conv(out)
#
#         disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
#         # disp = disp.squeeze(1)  # [B, H, W]
#
#         return disp


# class StereoDRNetRefinement(nn.Module):
#     def __init__(self):
#         super(StereoDRNetRefinement, self).__init__()
#
#         # Left and warped error
#         in_channels = 6
#
#         self.conv1 = conv2d(in_channels, 16)
#         self.conv2 = conv2d(1, 16)  # on low disparity
#
#         self.dilation_list = [1, 2, 4, 8, 1, 1]
#         self.dilated_blocks = nn.ModuleList()
#
#         for dilation in self.dilation_list:
#             self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))
#
#         self.dilated_blocks = nn.Sequential(*self.dilated_blocks)
#
#         self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)
#
#     def forward(self, low_disp, left_img, right_img):
#         assert low_disp.dim() == 3
#         low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
#         scale_factor = left_img.size(-1) / low_disp.size(-1)
#         if scale_factor == 1.0:
#             disp = low_disp
#         else:
#             disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
#             disp = disp * scale_factor
#
#         # Warp right image to left view with current disparity
#         warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
#         error = warped_right - left_img  # [B, C, H, W]
#
#         concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]
#
#         conv1 = self.conv1(concat1)  # [B, 16, H, W]
#         conv2 = self.conv2(disp)  # [B, 16, H, W]
#         concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]
#
#         out = self.dilated_blocks(concat2)  # [B, 32, H, W]
#         residual_disp = self.final_conv(out)  # [B, 1, H, W]
#
#         disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
#         disp = disp.squeeze(1)  # [B, H, W]
#
#         return disp


class HourglassRefinement(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(HourglassRefinement, self).__init__()
        self.num = 0;
        # Left and warped error
        in_channels = 6
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.conv_start = BasicConv(32, 32, kernel_size=3, padding=2, dilation=2)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, depth_map, ref_img, warped_imgs):
        assert depth_map.dim() == 4


        # print(cv2.imwrite(
        #     '/home/god/mvs/IterMVS-main_lstm_mutiscale_refiment_no_is_refine_new_aanet_copy/depth_refinement.jpg',
        #     255 * depth_map.view(1, ref_img.shape[3], ref_img.shape[4]).permute(1, 2, 0).cpu().numpy()))
        # exit()
        # Warp src image to ref view with current depth
        max_son = torch.max(depth_map)
        min_son = torch.min(depth_map)
        # avg_son = torch.mean(depth_map)

        # data = depth_map.flatten().detach().cpu().numpy()
        # plt.hist(data)
        # name = './img_dtu%s.jpg' % (self.num)
        # print(name)
        # plt.savefig(name)
        # self.num = self.num + 1
        # depth_map = ((depth_map - min_son) / (max_son - min_son)) * 1 #blend_test_tank
        depth_map = ((depth_map - min_son) / (max_son - min_son)) * 800 + 400 #dtu_test_tank
        num_views = ref_img.shape[1]
        sum_RGB = torch.sum(warped_imgs, dim=2, keepdims=True)
        # print(sum)
        ref_img_repeat = ref_img.repeat(1, num_views, 1, 1, 1)
        # print(ref_img.shape)
        # print(sum_RGB.shape)
        # print(warped_imgs.shape)

        warped_imgs_new = torch.where((sum_RGB == 3),
                                      ref_img_repeat, warped_imgs)

        # print(warped_imgs_new.shape)
        error = warped_imgs_new - ref_img  # [B, V, C, H, W]
        # exit()
        error = torch.sum(error, dim=1, keepdims=False)  # [B, C, H, W]
        ref_img = torch.squeeze(ref_img, dim=1)
        concat1 = torch.cat((error, ref_img), dim=1)  # [B, 6, H, W]
        conv1 = self.conv1(concat1)  # [B, 16, H, W]e
        conv2 = self.conv2(depth_map)  # [B, 16, H, W]
        x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        residual_disp = self.final_conv(x)  # [B, 1, H, W]

        depth_map = F.relu(depth_map + residual_disp, inplace=True)  # [B, 1, H, W]
        # depth_map = (max_son - min_son) * (depth_map) + min_son #blend_test_tank
        depth_map = (max_son - min_son) * (depth_map - 400) / 800 + min_son #dtu_test_tank

        return depth_map
