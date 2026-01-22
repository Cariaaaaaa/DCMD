import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PartialModalityCompensation(nn.Module):
    """
    部分模态补偿模块
    优化点：1. 滑动窗口批量计算局部方差；2. 增强注意力机制；3. 减少CPU操作
    """
    
    def __init__(self, feature_channels=[256, 512, 1024]):
        super(PartialModalityCompensation, self).__init__()
        
        self.feature_channels = feature_channels
        
        # 跨模态注意力模块（增强版）
        self.cross_modal_attentions = nn.ModuleList([
            EnhancedCrossModalAttention(channels) for channels in feature_channels
        ])
        
        # 无效区域检测参数
        self.depth_invalid_threshold = 0.0
        self.rgb_blur_threshold = 50.0  # 拉普拉斯方差阈值
        self.window_size = 8  # 滑动窗口大小
        self.stride = 4       # 滑动窗口步长
        
        # 预定义拉普拉斯算子
        self.laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', self.laplacian_kernel)
    
    def forward(self, rgb_features, depth_features, rgb_img, depth_img):
        """部分模态补偿（保持原有接口）"""
        batch_size = rgb_img.size(0)
        
        # 检测无效区域（批量滑动窗口计算）
        rgb_invalid_masks = self.detect_rgb_invalid_regions(rgb_img)
        depth_invalid_masks = self.detect_depth_invalid_regions(depth_img)
        
        # 多尺度特征补偿
        rgb_features_comp = []
        depth_features_comp = []
        
        for i, (rgb_feat, depth_feat, attention) in enumerate(zip(
            rgb_features, depth_features, self.cross_modal_attentions)):
            
            # 获取当前尺度的无效掩码
            scale = rgb_feat.size(2) / rgb_img.size(2)
            rgb_invalid = F.interpolate(rgb_invalid_masks.unsqueeze(1), 
                                      size=rgb_feat.shape[2:], 
                                      mode='nearest').squeeze(1)
            depth_invalid = F.interpolate(depth_invalid_masks.unsqueeze(1), 
                                        size=depth_feat.shape[2:], 
                                        mode='nearest').squeeze(1)
            
            # 特征补偿
            rgb_compensated = self.compensate_features(
                rgb_feat, depth_feat, rgb_invalid, attention, 'rgb2depth')
            depth_compensated = self.compensate_features(
                depth_feat, rgb_feat, depth_invalid, attention, 'depth2rgb')
            
            rgb_features_comp.append(rgb_compensated)
            depth_features_comp.append(depth_compensated)
        
        return rgb_features_comp, depth_features_comp
    
    def detect_rgb_invalid_regions(self, rgb_img):
        """
        检测RGB无效区域（模糊区域）
        优化：使用Unfold实现批量滑动窗口计算
        """
        batch_size, channels, height, width = rgb_img.shape
        
        # 转换为灰度图
        rgb_gray = 0.299 * rgb_img[:, 0] + 0.587 * rgb_img[:, 1] + 0.114 * rgb_img[:, 2]  # [B, H, W]
        rgb_gray = (rgb_gray * 255).clamp(0, 255)  # 缩放到0-255
        rgb_gray = rgb_gray.unsqueeze(1)  # [B, 1, H, W]
        
        # 计算拉普拉斯边缘
        laplace = F.conv2d(rgb_gray, self.laplacian, padding=1)  # [B, 1, H, W]
        
        # 滑动窗口提取（批量操作）
        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.stride)
        windows = unfold(laplace)  # [B, 1*window_size^2, N]，N为窗口数量
        windows = windows.permute(0, 2, 1)  # [B, N, window_size^2]
        
        # 计算每个窗口的方差
        window_vars = torch.var(windows, dim=2)  # [B, N]
        
        # 生成无效掩码
        invalid_masks = torch.zeros(batch_size, height, width, device=rgb_img.device)
        # 计算窗口坐标
        h_steps = (height - self.window_size) // self.stride + 1
        w_steps = (width - self.window_size) // self.stride + 1
        
        for b in range(batch_size):
            var_idx = 0
            for y in range(h_steps):
                for x in range(w_steps):
                    if window_vars[b, var_idx] < self.rgb_blur_threshold:
                        y_start = y * self.stride
                        y_end = y_start + self.window_size
                        x_start = x * self.stride
                        x_end = x_start + self.window_size
                        invalid_masks[b, y_start:y_end, x_start:x_end] = 1
                    var_idx += 1
        
        return invalid_masks
    
    def detect_depth_invalid_regions(self, depth_img):
        """检测深度无效区域（保持逻辑，优化代码）"""
        batch_size, channels, height, width = depth_img.shape
        
        # 提取深度通道
        depth_map = depth_img[:, 2, :, :] if channels == 3 else depth_img[:, 0, :, :]
        
        # 深度值<=阈值的区域标记为无效
        invalid_masks = (depth_map <= self.depth_invalid_threshold).float()  # [B, H, W]
        return invalid_masks
    
    def compensate_features(self, main_feat, aux_feat, invalid_mask, attention, direction):
        """特征补偿（保持逻辑）"""
        if invalid_mask.sum() == 0:
            return main_feat
        
        compensated_feat = attention(main_feat, aux_feat, invalid_mask, direction)
        
        # 混合原特征和补偿特征
        batch_size = main_feat.shape[0]
        result = main_feat.clone()
        for b in range(batch_size):
            mask = invalid_mask[b].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            mask = F.interpolate(mask, size=main_feat.shape[2:], mode='nearest')
            result[b] = main_feat[b] * (1 - mask) + compensated_feat[b] * mask
        
        return result


class EnhancedCrossModalAttention(nn.Module):
    """
    增强版跨模态注意力模块
    优化点：1. 加入位置编码；2. 改进注意力融合方式
    """
    
    def __init__(self, channels):
        super(EnhancedCrossModalAttention, self).__init__()
        
        self.channels = channels
        
        # 空间注意力（加入位置信息）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False),  # 4=2(avg/max)+2(位置编码)
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 特征变换与对齐
        self.feature_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, main_feat, aux_feat, invalid_mask, direction):
        batch_size, channels, height, width = main_feat.shape
        
        # 特征对齐
        aux_feat = self.feature_transform(aux_feat)
        
        # 生成位置编码（归一化坐标）
        x_coord = torch.linspace(-1, 1, width, device=main_feat.device).view(1, 1, 1, width).repeat(batch_size, 1, height, 1)
        y_coord = torch.linspace(-1, 1, height, device=main_feat.device).view(1, 1, height, 1).repeat(batch_size, 1, 1, width)
        pos_encoding = torch.cat([x_coord, y_coord], dim=1)  # [B, 2, H, W]
        
        # 空间注意力（融合特征统计量和位置编码）
        avg_pool = torch.mean(main_feat, dim=1, keepdim=True)
        max_pool, _ = torch.max(main_feat, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool, pos_encoding], dim=1)  # [B, 4, H, W]
        spatial_weights = self.spatial_attention(spatial_input)
        
        # 通道注意力
        channel_weights = self.channel_attention(main_feat)
        
        # 应用注意力权重（增强版：加入无效区域引导）
        invalid_mask_feat = F.interpolate(invalid_mask.unsqueeze(1).float(), 
                                         size=(height, width), mode='nearest')  # [B, 1, H, W]
        attended_aux = aux_feat * spatial_weights * channel_weights * (1 + invalid_mask_feat)  # 无效区域权重增强
        
        return attended_aux