import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from feature_fetcher import FeatureFetcher
from refinement import *
import cv2


class DepthInitialization(nn.Module):
    def __init__(self, num_sample):
        super(DepthInitialization, self).__init__()
        self.num_sample = num_sample

    def forward(self, inverse_depth_min, inverse_depth_max, height, width, device):
        batch = inverse_depth_min.size()[0]
        index = torch.arange(0, self.num_sample, 1, device=device).view(1, self.num_sample, 1, 1).float()
        normalized_sample = index.repeat(batch, 1, height, width) / (self.num_sample - 1)
        depth_sample = inverse_depth_max + normalized_sample * (inverse_depth_min - inverse_depth_max)

        depth_sample = 1.0 / depth_sample

        return depth_sample


class Evaluation(nn.Module):
    '''
    compute the correlation of all depth samples for each pixel
    '''

    def __init__(self):
        super(Evaluation, self).__init__()
        self.G = 8
        self.pixel_view_weight = PixelViewWeight(self.G)
        # correlation net for level 1,2,3
        self.corr_conv1 = nn.ModuleList([CorrNet(self.G), CorrNet(self.G), CorrNet(self.G)])

    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_sample, inverse_depth_min=None,
                inverse_depth_max=None, view_weights=None):
        V = len(src_features["level2"]) + 1

        if view_weights == None:
            correlation_sum = 0
            view_weight_sum = 1e-5
            view_weights = []
            batch, dim, height, width = ref_feature["level3"].size()

            ref_feature = ref_feature["level3"]
            src_features = src_features["level3"]

            ref_proj = ref_proj["level3"]
            num_sample = depth_sample.size()[1]

            for src_feature, src_proj in zip(src_features, src_projs["level3"]):
                warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_sample)  # [B,C,N,H,W]
                warped_feature = warped_feature.view(batch, self.G, dim // self.G, num_sample, height, width)
                correlation = torch.mean(
                    warped_feature * ref_feature.view(batch, self.G, dim // self.G, 1, height, width), dim=2,
                    keepdim=False)  # [B,G,N,H,W]

                view_weight = self.pixel_view_weight(correlation)  # [B,1,H,W]

                del warped_feature, src_feature, src_proj
                view_weights.append(F.interpolate(view_weight,
                                                  scale_factor=2, mode='bilinear'))

                if self.training:
                    correlation_sum = correlation_sum + correlation * view_weight.unsqueeze(1)  # [B, N, H, W]
                    view_weight_sum = view_weight_sum + view_weight.unsqueeze(1)  # [B,N,H,W]
                else:
                    correlation_sum += correlation * view_weight.unsqueeze(1)
                    view_weight_sum += view_weight.unsqueeze(1)
                del correlation, view_weight
            del src_features, src_projs

            # aggregated matching cost across all the source views
            correlation = correlation_sum.div_(view_weight_sum)  # [B,G,N,H,W]
            correlation = self.corr_conv1[-1](correlation)
            view_weights = torch.cat(view_weights, dim=1)  # .detach()
            del correlation_sum, view_weight_sum

            probability = torch.softmax(correlation, dim=1)
            # del correlation
            index = torch.arange(0, num_sample, 1, device=correlation.device).view(1, num_sample, 1, 1).float()
            index = torch.sum(index * probability, dim=1, keepdim=True)  # [B,1,H,W]
            normalized_depth = index / (num_sample - 1.0)
            depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
            depth = F.interpolate(depth,
                                  scale_factor=2, mode='bilinear')
            return view_weights, correlation, depth

        else:
            correlations = []
            for l in range(1, 4):
                correlation_sum = 0
                view_weight_sum = 1e-5
                ref_feature_l = ref_feature[f"level{l}"]
                ref_proj_l = ref_proj[f"level{l}"]
                depth_sample_l = depth_sample[f"level{l}"]
                batch, num_sample, height, width = depth_sample_l.size()
                dim = ref_feature_l.size(1)

                if not l == 2:
                    # need to interpolate
                    ref_feature_l = F.interpolate(ref_feature_l,
                                                  scale_factor=2 ** (l - 2), mode='bilinear')

                i = 0
                for src_feature, src_proj in zip(src_features[f"level{l}"], src_projs[f"level{l}"]):
                    warped_feature = differentiable_warping(src_feature, src_proj, ref_proj_l,
                                                            depth_sample_l)  # [B,C,N,H,W]
                    warped_feature = warped_feature.view(batch, self.G, dim // self.G, num_sample, height, width)
                    correlation = torch.mean(
                        warped_feature * ref_feature_l.view(batch, self.G, dim // self.G, 1, height, width), dim=2,
                        keepdim=False)  # [B,G,N,H,W]
                    view_weight = view_weights[:, i].view(batch, 1, 1, height, width)  # [B,1,H,W]

                    i = i + 1
                    del warped_feature, src_feature, src_proj

                    if self.training:
                        correlation_sum = correlation_sum + correlation * view_weight  # [B, N, H, W]
                        view_weight_sum = view_weight_sum + view_weight  # [B,1,H,W]

                    else:
                        correlation_sum += correlation * view_weight
                        view_weight_sum += view_weight
                    del correlation

                # aggregated matching cost across all the source views
                correlation = correlation_sum.div_(view_weight_sum)  # [B,G,N,H,W]
                correlation = self.corr_conv1[l - 1](correlation)
                correlations.append(correlation)
                del correlation_sum, correlation
            correlations = torch.cat(correlations, dim=1)

            return correlations


class Update(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_sample):
        super(Update, self).__init__()
        self.G = 4
        self.hidden_dim = hidden_dim
        self.out_num_samples = 256
        self.radius = 4

        # self.gru = ConvGRU(hidden_dim, input_dim)
        # self.lstm = LSTM(hidden_dim, input_dim)
        self.lstm04 = LSTM(32, 32 + 11)
        self.lstm08 = LSTM(32, 32 + 32)
        self.lstm16 = LSTM(32, 32)

        self.depth_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.out_num_samples, 1, stride=1, padding=0, dilation=1),
        )

        self.confidence_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, stride=1, padding=0, dilation=1),
        )

        self.hidden_init_head = nn.Sequential(
            nn.Conv2d(num_sample, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 1, stride=1, padding=0, dilation=1),
        )

    def hidden_init(self, corr):
        hidden = self.hidden_init_head(corr)
        hidden = F.interpolate(hidden,
                               scale_factor=2, mode='bilinear')
        hidden = torch.tanh(hidden)
        return hidden

    def conf_init(self, hidden):
        confidence_0 = self.confidence_head(hidden)
        confidence = torch.sigmoid(confidence_0)
        return confidence, confidence_0

    def depth_init(self, hidden):
        probability = torch.softmax(self.depth_head(hidden), dim=1)
        with torch.no_grad():
            index = torch.argmax(probability, dim=1, keepdim=True).type(torch.float)
            index_low = index - self.radius
            index = torch.arange(0, 2 * self.radius + 1, 1, device=hidden.device).view(1, 2 * self.radius + 1, 1,
                                                                                       1).float()
            index = index_low + index
            index = torch.clamp(index, min=0, max=self.out_num_samples - 1)
            index = index.type(torch.long)

        regress_index = 0
        probability_sum = 1e-6
        for i in range(2 * self.radius + 1):
            probability_1 = torch.gather(probability, 1, index[:, i:i + 1])
            regress_index = regress_index + index[:, i:i + 1] * probability_1
            probability_sum = probability_sum + probability_1
        regress_index = regress_index.div_(probability_sum)

        normalized_depth = regress_index / (self.out_num_samples - 1.0)
        return normalized_depth, probability

    def pool2x(self, x):
        return F.avg_pool2d(x, 3, stride=2, padding=1)

    def interp(self, x, dest):
        interp_args = {'mode': 'bilinear', 'align_corners': True}
        return F.interpolate(x, dest.shape[2:], **interp_args)

    def forward(self, h_list, c_list, normalized_depth, corr, confidence=None, confidence_flag=False):
        h04, h08, h16 = h_list
        c04, c08, c16 = c_list
        motion_features = torch.cat([normalized_depth, corr], dim=1)
        # print('motion_features')
        # print(motion_features.shape)
        # print('h16')
        # print(h16.shape)
        # print('h08.shape')
        # print(h08.shape)
        # print('h04.shape')
        # print(h04.shape)
        # exit()
        h16, c16 = self.lstm16(h16, c16, self.pool2x(h08))
        # print('h08')
        # print(h08.shape)
        h08, c08 = self.lstm08(h08, c08, self.pool2x(h04), self.interp(h16, h08))
        h04, c04 = self.lstm04(h04, c04, motion_features, self.interp(h08, h04))
        confidence_new_0 = None
        confidence_new = None
        if confidence_flag:
            confidence_new_0 = self.confidence_head(h04)
            confidence_new = torch.sigmoid(confidence_new_0)

        probability = torch.softmax(self.depth_head(h04), dim=1)

        with torch.no_grad():
            index = torch.argmax(probability, dim=1, keepdim=True).type(torch.float)
            index_low = index - self.radius
            index = torch.arange(0, 2 * self.radius + 1, 1, device=h04.device).view(1, 2 * self.radius + 1, 1,
                                                                                    1).float()
            index = index_low + index
            index = torch.clamp(index, min=0, max=self.out_num_samples - 1)
            index = index.type(torch.long)

        regress_index = 0
        probability_sum = 1e-6
        for i in range(2 * self.radius + 1):
            probability_1 = torch.gather(probability, 1, index[:, i:i + 1])
            regress_index = regress_index + index[:, i:i + 1] * probability_1
            probability_sum = probability_sum + probability_1
        regress_index = regress_index.div_(probability_sum)

        normalized_depth = regress_index / (self.out_num_samples - 1.0)
        h_list = (h04, h08, h16)
        c_list = (c04, c08, c16)
        return h_list, c_list, normalized_depth, probability, confidence_new, confidence_new_0


class IterMVS(nn.Module):
    def __init__(self, iteration, feature_dim, hidden_dim, test=False):
        super(IterMVS, self).__init__()

        # self.feature_grad_fetcher = FeatureGradFetcher()  # step
        # self.point_grad_fetcher = PointGrad()  # step
        self.refinement = HourglassRefinement()
        self.iteration = iteration
        self.hidden_dim = hidden_dim
        self.corr_sample = 10
        self.interval_scale = 1.0 / 256
        self.layer4 = self._make_layer(32, stride=2)
        self.layer5 = self._make_layer(32, stride=2)

        self.corr_interval = {
            "level1": torch.FloatTensor([-2, -2.0 / 3, 2.0 / 3, 2]).view(1, 4, 1, 1),
            "level2": torch.FloatTensor([-8, -8.0 / 3, 8.0 / 3, 8]).view(1, 4, 1, 1),
            "level3": torch.FloatTensor([-32, 32]).view(1, 2, 1, 1)
        }

        self.num_sample = 32
        self.test = test

        self.depth_initialization = DepthInitialization(self.num_sample)

        self.evaluation = Evaluation()

        self.update = Update(1 + 10, hidden_dim, self.num_sample)

        self.upsample = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16 * 9, 1, stride=1, padding=0, dilation=1, bias=False)
        )

    def gn_update(self, estimated_depth_map, isTest, extrinsics, intrinsics, img_tensor, n_views):  # 高斯牛顿法
        num_view = n_views[0].cpu().item()
        R = extrinsics[:, :, :3, :3]
        t = extrinsics[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        # print('num_view')
        # print(num_view)
        # exit()
        # print(estimated_depth_map.size(), image_scale)
        flow_height, flow_width = list(estimated_depth_map.size())[2:]
        # if flow_height != int(img_height * image_scale):
        #     flow_height = int(img_height * image_scale)
        #     flow_width = int(img_width * image_scale)
        #     estimated_depth_map = F.interpolate(estimated_depth_map, (flow_height, flow_width), mode="nearest")
        # else:
        #     # if it is the same size return directly
        #     return estimated_depth_map
        #     # pass

        if isTest:
            estimated_depth_map = estimated_depth_map.detach()

        # GN step
        # if isTest:
        #     cam_intrinsic[:, :, :2, :3] *= image_scale
        # else:
        #     cam_intrinsic[:, :, :2, :3] *= (4 * image_scale)
        batch_size = intrinsics.shape[0]
        ref_cam_intrinsic = intrinsics[:, 0:1, :, :].view(batch_size, 3, 3)
        # print(ref_cam_intrinsic)
        # exit()
        feature_map_indices_grid = self.get_pixel_grids(flow_height, flow_width) \
            .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(estimated_depth_map.device)
        # print('feature_map_indices_grid')
        # print(feature_map_indices_grid.shape)

        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1),
                          feature_map_indices_grid)  # (B, 1, 3, FH*FW)
        # print('torch.inverse(ref_cam_intrinsic).unsqueeze(1)')
        # print(torch.inverse(ref_cam_intrinsic).unsqueeze(1).shape)

        interval_depth_map = estimated_depth_map


        cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
            .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)
        # print('world_points')
        # print(world_points.shape)
        # exit()

        feature_fetcher = FeatureFetcher()  # step

        point_features = \
            feature_fetcher(img_tensor, world_points, intrinsics, extrinsics)

        #*******************************************************************************************
        # point_features = (point_features + 1) * 255
        # print('point_features')
        # print(point_features)
        # print(point_features[0:1, 1:2, :, :].shape)
        # img_tensor = (img_tensor + 1) * 255
        # point_features = point_features[0:1, 1:2, :, :].view(3, flow_height, flow_width).permute(1, 2, 0)
        # img_tensor = img_tensor[0:1, 0:1, :, :].view(3, flow_height, flow_width).permute(1, 2, 0)
        # point_features = point_features/2
        # img_tensor = img_tensor-255/2
        # point_features = (point_features + 1) * 255
        # img_tensor = (img_tensor + 1) * 255
        # print(img_tensor.cpu().numpy().shape)
        #
        # print(point_features.cpu().numpy().shape)
        # print(cv2.imwrite('/home/god/mvs/Miper-MVS_400_1200_copy_best2/ref_yuantu.jpg',
        #                   img_tensor.cpu().numpy()))
        # print(cv2.imwrite('/home/god/mvs/Miper-MVS_400_1200_copy_best2/ref_wrap.jpg',
        #                   point_features.cpu().numpy()))
        # exit()
        # *******************************************************************************************
        ref_img = img_tensor[:, 0:1, :, :].view(batch_size, 1, 3, flow_height, flow_width)

        num_view = point_features.shape[1]
        warped_img = point_features[:, :, :, :].view(batch_size, num_view, 3, flow_height, flow_width)
        # print(estimated_depth_map*1000)
        # exit()

        depth_map = self.refinement(interval_depth_map, ref_img, warped_img)
        # print(interval_depth_map)
        # print(cv2.imwrite('/home/god/mvs/IterMVS-main_lstm_mutiscale_refiment_no_is_refine_new_aanet_copy/depth_refinement.jpg',
        #                   255*depth_map.view(1, flow_height, flow_width).permute(1, 2, 0).cpu().numpy()))
        # print(cv2.imwrite(
        #     '/home/god/mvs/IterMVS-main_lstm_mutiscale_refiment_no_is_refine_new_aanet_copy/depth.jpg',
        #     255*interval_depth_map.view(1, flow_height, flow_width).permute(1, 2, 0).cpu().numpy()))
        # print(depth_map)
        # exit()
        # print(estimated_depth_map)
        # exit()
        # ************ second projection ************ #


        return depth_map

    def get_pixel_grids(self, height, width):
        with torch.no_grad():
            # texture coordinate
            x_linspace = torch.linspace(0.5, width - 0.5, width).view(1, width).expand(height, width)
            y_linspace = torch.linspace(0.5, height - 0.5, height).view(height, 1).expand(height, width)
            # y_coordinates, x_coordinates = torch.meshgrid(y_linspace, x_linspace)
            x_coordinates = x_linspace.contiguous().view(-1)
            y_coordinates = y_linspace.contiguous().view(-1)
            ones = torch.ones(height * width)
            indices_grid = torch.stack([x_coordinates, y_coordinates, ones], dim=0)
        return indices_grid

    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_min, depth_max, extrinsics, intrinsics,
                imgs, n_views):
        depths = {"combine": [], "probability": [], "initial": []}
        confidences = []
        depths_upsampled = []
        confidence_upsampled = None

        device = ref_feature["level2"].device
        batch, _, height, width = ref_feature["level2"].size()

        upsample_weight = self.upsample(ref_feature["level2"])  # [B,16*9,H,W]
        upsample_weight = upsample_weight.view(batch, 1, 9, 4, 4, height, width)
        upsample_weight = torch.softmax(upsample_weight, dim=2)

        inverse_depth_min = (1.0 / depth_min).view(batch, 1, 1, 1)
        inverse_depth_max = (1.0 / depth_max).view(batch, 1, 1, 1)

        depth_samples = self.depth_initialization(inverse_depth_min, inverse_depth_max, height // 2, width // 2, device)
        view_weights, corr, depth = self.evaluation(ref_feature, src_features, ref_proj, src_projs, depth_samples,
                                                    inverse_depth_min, inverse_depth_max)  # [B,2r+1,H,W]
        if not self.test:
            depths["initial"].append(depth)
        h04 = self.update.hidden_init(corr)
        h08 = self.layer4(h04)
        h16 = self.layer5(h08)
        h_list = (h04, h08, h16)
        c04, c08, c16 = h04, h08, h16
        c_list = (c04, c08, c16)
        normalized_depth, probability = self.update.depth_init(h04)

        if not self.test:
            confidence, confidence_0 = self.update.conf_init(h04)
            depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
            depths["combine"].append(depth)
            depths["probability"].append(probability)

            confidences.append(confidence_0)
            confidence = confidence.detach()
            normalized_depth = normalized_depth.detach()

        for iter in range(self.iteration):
            samples = {}
            for i in range(1, 4):
                normalized_sample = normalized_depth + self.corr_interval[f"level{i}"].to(
                    device) * self.interval_scale  # [B,R,H,W]
                normalized_sample = torch.clamp(normalized_sample, min=0, max=1)
                samples[f"level{i}"] = depth_unnormalization(normalized_sample, inverse_depth_min, inverse_depth_max)

            corr = self.evaluation(ref_feature, src_features, ref_proj, src_projs, samples,
                                   view_weights=view_weights.detach())  # [B,2r+1,H,W]

            if not self.test:
                h_list, c_list, normalized_depth, probability, confidence, confidence_0 = self.update(h_list, c_list,
                                                                                                      normalized_depth,
                                                                                                      corr,
                                                                                                      confidence_flag=True)

                depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
                depths["combine"].append(depth)
                depths["probability"].append(probability)

                confidences.append(confidence_0)

                if iter == self.iteration - 1:
                    depth_upsampled = upsample(normalized_depth, upsample_weight)
                    depth_upsampled = depth_unnormalization(depth_upsampled, inverse_depth_min, inverse_depth_max)

                    depth_upsampled2 = self.gn_update(depth_upsampled, False, extrinsics, intrinsics, imgs, n_views
                                                     )

                    depths_upsampled.append(depth_upsampled)
                    depths_upsampled.append(depth_upsampled2)
                    confidence_upsampled = F.interpolate(confidence,
                                                         scale_factor=4, mode='bilinear')

                confidence = confidence.detach()
                normalized_depth = normalized_depth.detach()
            else:
                if iter < self.iteration - 1:
                    h_list, c_list, normalized_depth, _, _, _ = self.update(h_list, c_list, normalized_depth, corr,
                                                                            confidence_flag=False)
                else:
                    depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
                    h_list, c_list, normalized_depth, _, confidence, _ = self.update(h_list, c_list, normalized_depth,
                                                                                     corr,
                                                                                     confidence_flag=True)
                    depth_upsampled = upsample(normalized_depth, upsample_weight)
                    depth_upsampled = depth_unnormalization(depth_upsampled, inverse_depth_min, inverse_depth_max)

                    depth_upsampled2 = self.gn_update(depth_upsampled, True, extrinsics, intrinsics, imgs, n_views
                                                     )
                    confidence_upsampled = F.interpolate(confidence,
                                                         scale_factor=4, mode='bilinear')


        if self.test:
            return depth, depth_upsampled, depth_upsampled2, confidence, confidence_upsampled
        else:
            return depths, depths_upsampled, confidences, confidence_upsampled

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(dim, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)


# estimate pixel-wise view weight
class PixelViewWeight(nn.Module):
    def __init__(self, G):
        super(PixelViewWeight, self).__init__()
        self.conv = nn.Sequential(
            ConvReLU(G, 16),
            nn.Conv2d(16, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        # x: [B, G, N, H, W]
        batch, dim, num_depth, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch * num_depth, dim, height, width)  # [B*N,G,H,W]
        x = self.conv(x).view(batch, num_depth, height, width)
        x = torch.softmax(x, dim=1)
        x = torch.max(x, dim=1)[0]

        return x.unsqueeze(1)


class CorrNet(nn.Module):
    def __init__(self, G):
        super(CorrNet, self).__init__()
        self.conv0 = ConvReLU(G, 8)
        self.conv1 = ConvReLU(8, 16, stride=2)
        self.conv2 = ConvReLU(16, 32, stride=2)

        self.conv3 = nn.ConvTranspose2d(32, 16, 3, padding=1, output_padding=1,
                                        stride=2, bias=False)

        self.conv4 = nn.ConvTranspose2d(16, 8, 3, padding=1, output_padding=1,
                                        stride=2, bias=False)

        self.conv5 = nn.Conv2d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        batch, dim, num_depth, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch * num_depth, dim, height, width)  # [B*N,G,H,W]
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        x = self.conv2(conv1)

        x = conv1 + self.conv3(x)
        del conv1
        x = conv0 + self.conv4(x)
        del conv0

        x = self.conv5(x).view(batch, num_depth, height, width)
        return x
