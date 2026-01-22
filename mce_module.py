import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class ModalityConfidenceModule(nn.Module):
    """
    模态置信度评估模块
    计算RGB和3D模态的置信度得分，用于自适应融合
    优化点：使用批量操作替代循环，减少CPU/GPU数据传输
    """
    
    def __init__(self):
        super(ModalityConfidenceModule, self).__init__()
        self.eps = 1e-4
        
        # 拉普拉斯算子卷积核（3x3）
        self.laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', self.laplacian_kernel)
        
        # 预计算训练数据的最大统计值（在实际使用中应从训练数据计算）
        self.register_buffer('max_laplace_var', torch.tensor(1000.0))
        self.register_buffer('max_density', torch.tensor(10000.0))
    
    def forward(self, rgb_img, depth_img):
        """计算模态置信度（保持原有接口）"""
        batch_size = rgb_img.size(0)
        
        # 计算RGB置信度（批量操作）
        rgb_confidence = self.compute_rgb_confidence(rgb_img)  # [B, 1]
        
        # 计算深度置信度（批量操作）
        depth_confidence = self.compute_depth_confidence(depth_img)  # [B, 1]
        
        # 自适应权重归一化
        w_r = rgb_confidence / (rgb_confidence + depth_confidence + self.eps)
        w_d = depth_confidence / (rgb_confidence + depth_confidence + self.eps)
        
        return w_r, w_d
    
    def compute_rgb_confidence(self, rgb_img):
        """
        计算RGB置信度（批量优化）
        基于模糊度（拉普拉斯方差）和光照均匀性（直方图熵）
        """
        batch_size, channels, height, width = rgb_img.shape
        
        # 转换为灰度图（批量操作）
        rgb_gray = 0.299 * rgb_img[:, 0] + 0.587 * rgb_img[:, 1] + 0.114 * rgb_img[:, 2]  # [B, H, W]
        rgb_gray = rgb_gray.unsqueeze(1)  # [B, 1, H, W]
        
        # 批量计算拉普拉斯方差（模糊度）
        laplacian = F.conv2d(rgb_gray, self.laplacian, padding=1)  # [B, 1, H, W]
        laplace_var = torch.var(laplacian, dim=(1, 2, 3))  # [B] 计算每个样本的方差
        
        # 批量计算直方图熵（光照均匀性）
        # 将像素值缩放到0-255并离散化
        rgb_gray_scaled = (rgb_gray * 255).clamp(0, 255).long()  # [B, 1, H, W]
        entropy = self.batch_histogram_entropy(rgb_gray_scaled, bins=256)  # [B]
        
        # 归一化得分
        clarity_score = laplace_var / (self.max_laplace_var + self.eps)
        uniformity_score = 1.0 / (entropy + self.eps)
        
        # 综合得分（可调整权重）
        rgb_score = 0.5 * (clarity_score + uniformity_score)
        return rgb_score.unsqueeze(1)
    
    def compute_depth_confidence(self, depth_img):
        """
        计算深度置信度（批量优化）
        基于点密度和噪声水平
        """
        batch_size, channels, height, width = depth_img.shape
        
        # 提取深度通道（批量操作）
        if channels == 3:
            depth_map = depth_img[:, 2, :, :]  # [B, H, W]
        else:
            depth_map = depth_img[:, 0, :, :]  # [B, H, W]
        
        # 批量计算点密度（有效深度点比例）
        valid_mask = depth_map > 0  # [B, H, W]
        density = valid_mask.float().mean(dim=(1, 2))  # [B] 每个样本的有效点比例
        
        # 批量计算噪声水平（有效区域标准差）
        noise_level = torch.zeros(batch_size, device=depth_img.device)
        for b in range(batch_size):
            valid_depths = depth_map[b][valid_mask[b]]
            if valid_depths.numel() > 0:
                noise_level[b] = torch.std(valid_depths)
            else:
                noise_level[b] = 1.0  # 无有效点时设为高噪声
        
        # 归一化得分
        density_score = density / (self.max_density + self.eps)
        noise_score = 1.0 / (noise_level + self.eps)
        
        # 综合得分
        depth_score = 0.5 * (density_score + noise_score)
        return depth_score.unsqueeze(1)
    
    def batch_histogram_entropy(self, x, bins=256):
        """批量计算直方图熵"""
        batch_size = x.shape[0]
        entropy = torch.zeros(batch_size, device=x.device)
        
        for b in range(batch_size):
            # 计算单个样本的直方图
            hist = torch.histc(x[b].float(), bins=bins, min=0, max=255)
            hist = hist / hist.sum()  # 归一化
            entropy[b] = -torch.sum(hist * torch.log2(hist + self.eps))
        
        return entropy
    
    def update_max_stats(self, rgb_imgs, depth_imgs):
        """更新训练数据的最大统计值（批量优化）"""
        # 计算RGB的拉普拉斯方差最大值
        rgb_gray = 0.299 * rgb_imgs[:, 0] + 0.587 * rgb_imgs[:, 1] + 0.114 * rgb_imgs[:, 2]
        rgb_gray = rgb_gray.unsqueeze(1)
        laplacian = F.conv2d(rgb_gray, self.laplacian, padding=1)
        laplace_vars = torch.var(laplacian, dim=(1, 2, 3))
        self.max_laplace_var = torch.max(self.max_laplace_var, laplace_vars.max())
        
        # 计算深度的点密度最大值
        if depth_imgs.size(1) == 3:
            depth_maps = depth_imgs[:, 2, :, :]
        else:
            depth_maps = depth_imgs[:, 0, :, :]
        valid_masks = depth_maps > 0
        densities = valid_masks.float().mean(dim=(1, 2))
        self.max_density = torch.max(self.max_density, densities.max())