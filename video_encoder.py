"""
视频JSCC编码器和解码器

基于DCVC的视频编码器，修改为JSCC架构。
移除显式量化和熵编码，支持连续值特征传输。
包含光流估计、运动补偿、上下文编码等时序模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import logging
from image_encoder import SNRModulator

logger = logging.getLogger(__name__)


def _log_nonfinite(name: str, tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor):
        return False
    finite_mask = torch.isfinite(tensor)
    if finite_mask.all():
        return False
    with torch.no_grad():
        finite_values = tensor[finite_mask]
        if finite_values.numel() > 0:
            t_min = finite_values.min().item()
            t_max = finite_values.max().item()
            t_mean = finite_values.mean().item()
        else:
            t_min = float("nan")
            t_max = float("nan")
            t_mean = float("nan")
    logger.warning(
        "%s has NaN/Inf values (finite stats min=%.6e max=%.6e mean=%.6e)",
        name,
        t_min,
        t_max,
        t_mean,
    )
    return True


class LightweightTemporalConv(nn.Module):
    """
    轻量级时序卷积模块（替代ConvLSTM）
    
    使用简单的时序卷积进行时序建模，显存占用远小于ConvLSTM。
    通过残差连接和门控机制保持时序建模能力。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        batch_first: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # 轻量级时序卷积层
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.GroupNorm(8, hidden_dim),  # 使用GroupNorm替代BN，更节省显存
                nn.ReLU(inplace=True)
            ))
        self.layers = nn.ModuleList(layers)
        
        # 门控机制（轻量级）
        self.gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_tensor (torch.Tensor): 输入张量 [B, T, C, H, W] 或 [B, 1, C, H, W]
            hidden_state (torch.Tensor, optional): 前一时刻的隐藏状态 [B, C, H, W]
            
        Returns:
            Tuple: (输出张量, 新的隐藏状态)
        """
        if not self.batch_first and input_tensor.dim() == 5:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        if input_tensor.dim() == 5:
            # [B, T, C, H, W] -> 取最后一帧
            input_tensor = input_tensor[:, -1]  # [B, C, H, W]
        
        # 如果提供了隐藏状态，进行融合
        if hidden_state is not None:
            # 门控融合
            combined = torch.cat([input_tensor, hidden_state], dim=1)
            gate_weight = self.gate(combined)
            input_tensor = input_tensor * gate_weight + hidden_state * (1 - gate_weight)
        
        # 通过时序卷积层
        x = input_tensor
        for layer in self.layers:
            x = layer(x)
        
        # 残差连接
        if self.input_dim == self.hidden_dim:
            x = x + input_tensor
        
        # 返回输出和新的隐藏状态
        return x, x







class LightweightOpticalFlow(nn.Module):
    """
    轻量级光流估计模块（简化版）
    
    使用更小的网络和简化的架构，大幅减少显存占用。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 32,  # 减小隐藏维度
        num_layers: int = 2  # 减少层数
    ):
        super().__init__()
        self.in_channels = in_channels
        
        # 简化的光流估计网络（2层）
        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=3, padding=1, stride=2),  # 下采样减少计算
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样回原尺寸
        )
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        估计光流（轻量级版本）
        
        Args:
            frame1 (torch.Tensor): 参考帧 [B, C, H, W]
            frame2 (torch.Tensor): 目标帧 [B, C, H, W]
            
        Returns:
            torch.Tensor: 光流 [B, 2, H, W]
        """
        # 拼接两帧
        input_tensor = torch.cat([frame1, frame2], dim=1)
        
        # 估计光流
        flow = self.flow_net(input_tensor)
        
        return flow


# 保持兼容性
class OpticalFlow(LightweightOpticalFlow):
    """兼容性包装"""
    pass


class MotionCompensation(nn.Module):
    """
    运动补偿模块
    
    基于光流进行运动补偿，生成预测帧。
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        reference_frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        运动补偿
        
        Args:
            reference_frame (torch.Tensor): 参考帧 [B, C, H, W]
            flow (torch.Tensor): 光流 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 补偿后的帧 [B, C, H, W]
        """
        B, C, H, W = reference_frame.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=reference_frame.device),
            torch.arange(W, device=reference_frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 应用光流
        flow_grid = grid + flow
        
        # 归一化到[-1, 1]
        flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
        flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # 重排列为grid_sample格式
        flow_grid = flow_grid.permute(0, 2, 3, 1)
        
        # 双线性插值
        compensated_frame = F.grid_sample(
            reference_frame, flow_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return compensated_frame


class ContextualEncoder(nn.Module):
    """
    上下文编码器
    
    基于DCVC的上下文编码器，移除量化模块，输出连续值特征。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # 特征提取层
        layers = []
        in_ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            if i < num_layers - 1:
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        上下文编码
        
        Args:
            x (torch.Tensor): 输入帧 [B, C, H, W]
            
        Returns:
            torch.Tensor: 编码特征 [B, hidden_dim, H', W']
        """
        features = self.encoder(x)
        features = self.output_proj(features)
        return features


class ContextualDecoder(nn.Module):
    """
    上下文解码器
    
    基于DCVC的上下文解码器，从连续值特征重建帧。
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 解码器层
        layers = []
        in_ch = hidden_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else out_channels
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Sigmoid()
            ])
            if i < num_layers - 1:
                layers.append(nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        上下文解码
        
        Args:
            x (torch.Tensor): 输入特征 [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: 重建帧 [B, out_channels, H', W']
        """
        x = self.input_proj(x)
        x = self.decoder(x)
        return x


class VideoJSCCEncoder(nn.Module):
    """
    视频JSCC编码器
    
    基于DCVC架构的视频编码器，修改为JSCC模式。
    移除量化和熵编码，支持连续值特征传输。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_frames: int = 5,
        use_optical_flow: bool = True,
        use_convlstm: bool = True,
        output_dim: int = 256,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        patch_embed: Optional[nn.Module] = None,
        swin_layers: Optional[nn.ModuleList] = None,
        swin_norm: Optional[nn.Module] = None,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.use_optical_flow = use_optical_flow
        self.use_convlstm = use_convlstm
        
        # 光流估计
        if use_optical_flow:
            self.optical_flow = OpticalFlow(in_channels, hidden_dim)
            self.motion_compensation = MotionCompensation()
        
        # 使用Swin Transformer作为视觉主干网（支持共享注入）
        if (patch_embed is not None) and (swin_layers is not None) and (swin_norm is not None):
            self.patch_embed = patch_embed
            self.swin_layers = swin_layers
            self.swin_norm = swin_norm
        else:
            from image_encoder import PatchEmbed, BasicLayer
            patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
            # Patch embedding（非共享路径）
            self.patch_embed = PatchEmbed(
                img_size=img_size,  # 假设输入尺寸
                patch_size=patch_size,
                in_chans=in_channels,
                embed_dim=hidden_dim
            )
            # Swin Transformer层（非共享路径）
            self.swin_layers = nn.ModuleList([
                BasicLayer(
                    dim=hidden_dim,
                    out_dim=hidden_dim,
                    input_resolution=patches_resolution,  # 224/4 = 56
                    depth=2,
                    num_heads=8,
                    window_size=4,
                    mlp_ratio=mlp_ratio,  # 修复OOM：从4.0减小到2.0以节省显存
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.1,
                    norm_layer=nn.LayerNorm,
                    downsample=None
                )
            ])
            self.swin_norm = nn.LayerNorm(hidden_dim)
        
        # 特征重塑层
        self.feature_reshape = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.snr_modulator = SNRModulator(hidden_dim)
        
        # 轻量级时序卷积建模（重构：使用LightweightTemporalConv替代ConvLSTM）
        if use_convlstm:
            self.temporal_layer = LightweightTemporalConv(  # 实际上是LightweightTemporalConv
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=3,
                num_layers=1  # 重构：减少层数以节省显存
            )
            self.hidden_state = None  # 重构：hidden_state现在是单个tensor而不是list
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, output_dim, kernel_size=1)
        )
        
        # 引导向量提取器
        self.guide_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        
        # 初始化隐藏状态
        self.hidden_state = None
    
    def forward(
        self,
        video_frames: torch.Tensor,
        reset_state: bool = False,
        snr_db: float = 10.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        视频编码器前向传播 - 特征空间运动补偿
        
        Args:
            video_frames (torch.Tensor): 视频帧 [B, T, C, H, W]
            reset_state (bool): 是否重置隐藏状态
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (编码特征, 引导向量)
        """
        B, T, C, H, W = video_frames.shape
        
        if reset_state or self.hidden_state is None:
            self.hidden_state = None
        
        encoded_features = []
        guide_vectors = []
        prev_feature = None  # 前一帧的特征
        prev_frame = None   # 前一帧的像素
        
        for t in range(T):
            current_frame = video_frames[:, t, :, :, :]
            
            # 使用Swin Transformer提取当前帧特征
            current_feature = self._extract_swin_features(current_frame)
            current_feature = self.snr_modulator(current_feature, snr_db)
            
            # 特征空间运动补偿
            if self.use_optical_flow and t > 0 and prev_feature is not None:
                # 在像素空间估计光流
                flow = self.optical_flow(prev_frame, current_frame)
                
                # 将像素空间光流对齐到特征空间尺寸，并按比例缩放位移
                feat_h, feat_w = prev_feature.shape[-2], prev_feature.shape[-1]
                img_h, img_w = current_frame.shape[-2], current_frame.shape[-1]
                flow_downsampled = F.interpolate(flow, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
                scale_w = float(feat_w) / float(img_w)
                scale_h = float(feat_h) / float(img_h)
                flow_rescaled = torch.zeros_like(flow_downsampled)
                flow_rescaled[:, 0] = flow_downsampled[:, 0] * scale_w
                flow_rescaled[:, 1] = flow_downsampled[:, 1] * scale_h
                
                # 使用重标定后的光流扭曲前一帧特征
                warped_feature = self._warp_features(prev_feature, flow_rescaled)
                
                # 计算特征残差
                feature_residual = current_feature - warped_feature
                
                # 轻量级时序卷积传播特征（重构：使用LightweightTemporalConv替代ConvLSTM）
                if self.use_convlstm:
                    # 新的LightweightTemporalConv接受[B, C, H, W]格式，不需要unsqueeze
                    feature_residual, self.hidden_state = self.temporal_layer(
                        feature_residual, self.hidden_state
                    )
                feature_residual = self.snr_modulator(feature_residual, snr_db)
                
                # 输出投影
                encoded_frame = self.output_proj(feature_residual)
                encoded_features.append(encoded_frame)
                
                # 提取引导向量
                guide_vector = self.guide_extractor(feature_residual)
                guide_vectors.append(guide_vector)
                
                # 更新前一帧特征（用于下一帧）
                prev_feature = current_feature
            else:
                # 第一帧或没有光流
                if self.use_convlstm:
                    # 新的LightweightTemporalConv接受[B, C, H, W]格式
                    current_feature, self.hidden_state = self.temporal_layer(
                        current_feature, self.hidden_state
                    )
                current_feature = self.snr_modulator(current_feature, snr_db)
                
                # 输出投影
                encoded_frame = self.output_proj(current_feature)
                encoded_features.append(encoded_frame)
                
                # 提取引导向量
                guide_vector = self.guide_extractor(current_feature)
                guide_vectors.append(guide_vector)
                
                # 更新前一帧特征
                prev_feature = current_feature
            
            # 更新前一帧像素（用于光流估计）
            if self.use_optical_flow:
                prev_frame = current_frame
        
        # 堆叠所有帧的特征
        encoded_features = torch.stack(encoded_features, dim=1)  # [B, T, C, H, W]
        guide_vectors = torch.stack(guide_vectors, dim=1)  # [B, T, guide_dim]
        _log_nonfinite("VideoJSCCEncoder.encoded_features", encoded_features)
        _log_nonfinite("VideoJSCCEncoder.guide_vectors", guide_vectors)

        return encoded_features, guide_vectors
    
    def _warp_features(self, features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        使用光流扭曲特征
        
        Args:
            features (torch.Tensor): 特征 [B, C, H, W]
            flow (torch.Tensor): 光流 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 扭曲后的特征
        """
        B, C, H, W = features.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=features.device),
            torch.arange(W, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 应用光流
        flow_grid = grid + flow
        
        # 归一化到[-1, 1]
        flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
        flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # 重排列为grid_sample格式
        flow_grid = flow_grid.permute(0, 2, 3, 1)
        
        # 双线性插值
        warped_features = F.grid_sample(
            features, flow_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped_features
    
    def _extract_swin_features(self, frame: torch.Tensor) -> torch.Tensor:
        """
        使用Swin Transformer提取帧特征
        
        Args:
            frame (torch.Tensor): 输入帧 [B, C, H, W]
            
        Returns:
            torch.Tensor: 提取的特征 [B, hidden_dim, H', W']
        """
        B, C, H, W = frame.shape
        
        # Patch embedding
        x = self.patch_embed(frame)  # [B, num_patches, hidden_dim]
        
        # 通过Swin Transformer层（鲁棒：若维度不兼容则跳过Swin层）
        #try:
        for layer in self.swin_layers:
            x = layer(x)
        #except Exception as e:
            #print(f"[Warn] VideoJSCCEncoder.swin_layers 失败，启用简化路径: {e}; x.shape={tuple(x.shape)}")
        
        # 层归一化
        if hasattr(self, 'swin_norm') and self.swin_norm is not None:
            try:
                x = self.swin_norm(x)
            except Exception as e:
                print(f"[Warn] VideoJSCCEncoder.swin_norm 失败（忽略）: {e}; x.shape={tuple(x.shape)}")
        
        # 重塑为特征图格式
        num_patches = x.shape[1]
        #patch_size = int(num_patches ** 0.5)
        #x = x.view(B, patch_size, patch_size, -1).permute(0, 3, 1, 2)  # [B, hidden_dim, H', W']
        H, W = frame.shape[-2:]
        feat_h = H // self.patch_embed.patch_size
        feat_w = W // self.patch_embed.patch_size
        assert num_patches == feat_h * feat_w, f"特征序列长度 {num_patches} 与计算出的网格 {feat_h}x{feat_w} 不匹配"
        x = x.view(B, feat_h, feat_w, -1).permute(0, 3, 1, 2)
        # 特征重塑
        x = self.feature_reshape(x)
        
        return x
    
    def reset_hidden_state(self):
        """重置隐藏状态"""
        self.hidden_state = None


class VideoJSCCDecoder(nn.Module):
    """
    视频JSCC解码器
    
    从经过信道传输的带噪特征重建原始视频序列。
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        hidden_dim: int = 256,
        num_frames: int = 5,
        use_optical_flow: bool = True,
        use_convlstm: bool = True,
        input_dim: int = 256,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        semantic_context_dim: int = 256,
        mlp_ratio: float = 4.0  # 添加语义上下文维度参数
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.use_optical_flow = use_optical_flow
        self.use_convlstm = use_convlstm
        self.img_size = img_size
        self.patch_size = patch_size
        self.semantic_context_dim = semantic_context_dim
        self.patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)  # (56, 56)
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1)
        )
        
        # 使用Swin Transformer作为视觉主干网
        from image_encoder import BasicLayer, PatchReverseMerging
        
        # 特征重塑层
        self.feature_reshape = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Swin Transformer层 - 添加上采样机制
        # 参考 ImageJSCCDecoder 的设计，使用多层 BasicLayer + PatchReverseMerging 逐步上采样
        # 从 56×56 -> 112×112 -> 224×224
        self.swin_layers = nn.ModuleList([
            # 第一层：处理 56×56 特征，输出仍为 56×56
            BasicLayer(
                dim=hidden_dim,
                out_dim=hidden_dim,
                input_resolution=self.patches_resolution,  # (56, 56)
                depth=2,
                num_heads=8,
                window_size=4,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=None  # 第一层不进行上采样
            ),
            # 第二层：上采样到 112×112
            BasicLayer(
                dim=hidden_dim,
                out_dim=hidden_dim,  # 保持维度不变
                input_resolution=self.patches_resolution,  # (56, 56) - 输入
                depth=2,
                num_heads=8,
                window_size=4,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=PatchReverseMerging  # 上采样：56×56 -> 112×112 (序列长度变为4倍)
            ),
            # 第三层：上采样到 224×224
            BasicLayer(
                dim=hidden_dim,
                out_dim=hidden_dim,  # 保持维度不变
                input_resolution=(self.patches_resolution[0] * 2, self.patches_resolution[1] * 2),  # (112, 112) - 输入
                depth=2,
                num_heads=8,
                window_size=4,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=PatchReverseMerging  # 上采样：112×112 -> 224×224 (序列长度变为4倍)
            )
        ])
        
        # 输出投影：使用 Conv2d 进行通道映射（因为上采样已在 Swin 层完成）
        # 由于 PatchReverseMerging 已经上采样到 224×224，这里仅进行通道映射
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.Sigmoid()  # 添加 Sigmoid 激活函数，确保输出在 [0, 1] 范围内
        )
        
        # 轻量级时序卷积建模（重构：使用LightweightTemporalConv替代ConvLSTM）
        if use_convlstm:
            self.temporal_layer = LightweightTemporalConv(  # 实际上是LightweightTemporalConv
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=3,
                num_layers=1  # 重构：减少层数以节省显存
            )
            self.hidden_state = None  # 重构：hidden_state现在是单个tensor而不是list
        
        # 光流估计（用于解码端）
        if use_optical_flow:
            self.optical_flow = OpticalFlow(out_channels, hidden_dim)
            self.motion_compensation = MotionCompensation()
            # 为像素域近似重建提供轻量级解码器，用于光流估计的当前帧输入
            self.contextual_decoder = ContextualDecoder(
                in_channels=hidden_dim,
                out_channels=out_channels,
                hidden_dim=hidden_dim
            )
        
        # 引导向量处理器
        self.guide_processor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 语义对齐层：在 __init__ 中预定义，而不是在 forward 中动态创建
        # 用于将语义上下文（文本编码）的维度对齐到视频特征的维度
        # 文本编码维度：semantic_context_dim (例如 256)
        # 视频特征维度：hidden_dim (例如 256)
        self.semantic_aligner = nn.Linear(semantic_context_dim, hidden_dim)
        
        # 【修复】使用标准CrossAttention模块替代简化的自定义注意力实现
        # 导入并使用标准CrossAttention模块（与ImageJSCCDecoder保持一致）
        from cross_attention import CrossAttention
        # CrossAttention 需要 embed_dim 参数，这里使用 hidden_dim（对齐后的维度）
        self.cross_attention = CrossAttention(
            embed_dim=hidden_dim,
            num_heads=max(1, hidden_dim // 64),  # 自适应头数：256//64=4
            dropout=0.0  # 可以使用dropout，这里设为0保持与原有实现一致
        )
        
        # 初始化隐藏状态
        self.hidden_state = None
        self.last_semantic_gate_stats: Dict[str, Optional[float]] = {"mean": None, "std": None}
    
    def forward(
        self,
        noisy_features: torch.Tensor,
        guide_vectors: torch.Tensor,
        reset_state: bool = False,
        semantic_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        视频解码器前向传播 - 特征空间运动补偿
        
        Args:
            noisy_features (torch.Tensor): 带噪特征 [B, T, C, H, W]
            guide_vectors (torch.Tensor): 引导向量 [B, T, guide_dim]
            reset_state (bool): 是否重置隐藏状态
            semantic_context (torch.Tensor, optional): 语义上下文 [B, seq_len, D_text]
            
        Returns:
            torch.Tensor: 重建视频 [B, T, C, H, W]
        """
        B, T, C, H, W = noisy_features.shape
        
        if reset_state or self.hidden_state is None:
            self.hidden_state = None
        
        decoded_frames = []
        prev_reconstructed_feature = None  # 前一帧重建的特征
        prev_decoded_frame = None         # 前一帧重建的像素
        device = noisy_features.device  # 保存设备信息，用于后续将CPU上的帧移回GPU
        
        for t in range(T):
            current_features = noisy_features[:, t, :, :, :]
            current_guide = guide_vectors[:, t, :]
            
            # 输入投影
            projected_features = self.input_proj(current_features)
            del current_features  # 及时释放
            
            # 处理引导向量
            guide_processed = self.guide_processor(current_guide)
            guide_expanded = guide_processed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            del current_guide, guide_processed  # 及时释放
            
            # 【重构】提前应用语义引导（在convlstm之前，与guide_expanded一起融合）
            if semantic_context is not None:
                # 应用语义引导的交叉注意力
                projected_features = self._apply_semantic_guidance(
                    projected_features, semantic_context
                )
                # 融合所有信息：特征 + 自身引导 + 语义引导
                projected_features = projected_features + guide_expanded
            else:
                # 仅融合自身引导
                projected_features = projected_features + guide_expanded
            del guide_expanded  # 及时释放
            
            # 轻量级时序卷积建模（重构：使用LightweightTemporalConv替代ConvLSTM）
            # 注意：语义引导已在convlstm之前应用，确保所有时序和语义信息在进入convlstm之前都已就位
            if self.use_convlstm:
                # 新的LightweightTemporalConv接受[B, C, H, W]格式
                projected_features, self.hidden_state = self.temporal_layer(
                    projected_features, self.hidden_state
                )
            
            # 特征空间运动补偿
            if self.use_optical_flow and t > 0 and prev_reconstructed_feature is not None:
                # 估计光流（在像素空间）
                # 优化：先解码 projected_features 得到预览帧用于光流估计
                preliminary_current_frame = self._decode_swin_features(projected_features)
                flow = self.optical_flow(prev_decoded_frame, preliminary_current_frame)
                
                # 将像素空间光流对齐到特征空间尺寸，并按比例缩放位移
                feat_h, feat_w = prev_reconstructed_feature.shape[-2], prev_reconstructed_feature.shape[-1]
                img_h, img_w = prev_decoded_frame.shape[-2], prev_decoded_frame.shape[-1]
                flow_downsampled = F.interpolate(flow, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
                del flow  # 及时释放
                
                scale_w = float(feat_w) / float(img_w)
                scale_h = float(feat_h) / float(img_h)
                flow_rescaled = torch.zeros_like(flow_downsampled)
                flow_rescaled[:, 0] = flow_downsampled[:, 0] * scale_w
                flow_rescaled[:, 1] = flow_downsampled[:, 1] * scale_h
                del flow_downsampled  # 及时释放
                
                # 扭曲前一帧特征
                warped_prev_feature = self._warp_features(prev_reconstructed_feature, flow_rescaled)
                del flow_rescaled  # 及时释放
                
                # 特征残差解码
                current_reconstructed_feature = warped_prev_feature + projected_features
                del warped_prev_feature  # 及时释放
                
                # 优化：复用已解码的 preview_current_frame，避免重复解码
                # 如果运动补偿后的特征与原始 projected_features 差异很小，说明运动补偿效果不明显
                # 此时可以复用预览帧，避免重复解码，显著提升效率
                feature_diff = torch.mean((current_reconstructed_feature - projected_features).abs())
                if feature_diff < 0.05:  # 特征差异阈值：如果差异很小，复用预览帧（节省一次完整解码）
                    decoded_frame = preliminary_current_frame
                    # 注意：如果复用预览帧，current_reconstructed_feature 仍然需要保留用于状态更新
                    # 但我们可以使用 projected_features 作为近似（因为差异很小）
                    current_reconstructed_feature = projected_features  # 使用近似值
                    del preliminary_current_frame  # 释放预览帧
                else:
                    # 差异较大，说明运动补偿有明显效果，需要重新解码以保持精度
                    del preliminary_current_frame  # 释放预览帧
                    decoded_frame = self._decode_swin_features(current_reconstructed_feature)
            else:
                # 第一帧或没有光流：直接解码
                current_reconstructed_feature = projected_features
                decoded_frame = self._decode_swin_features(current_reconstructed_feature)
            
            # 修复OOM：将解码帧移到CPU，最后再移回GPU（减少GPU峰值内存）
            # 保存解码帧（先移到CPU以节省GPU内存）
            if self.training:
                # 在移到CPU之前，先更新状态（如果需要）
                if t < T - 1:  # 不是最后一帧，需要保留状态
                    prev_reconstructed_feature = current_reconstructed_feature.detach().clone()  # 使用detach避免梯度累积
                    prev_decoded_frame = decoded_frame.detach().clone()  # 在移到CPU前克隆GPU上的帧
                # 移到CPU并释放GPU上的帧
                decoded_frames.append(decoded_frame.cpu())
                del decoded_frame  # 释放GPU上的帧
            else:
                # 推理时保持在GPU上
                if t < T - 1:  # 不是最后一帧，需要保留状态
                    prev_reconstructed_feature = current_reconstructed_feature.detach().clone()
                    prev_decoded_frame = decoded_frame.detach().clone()
                decoded_frames.append(decoded_frame)
            
        # 堆叠所有帧
        # 修复OOM：如果帧在CPU上，先移回GPU再堆叠
        if self.training and len(decoded_frames) > 0 and decoded_frames[0].device.type == 'cpu':
            # 将CPU上的帧移回GPU
            decoded_frames_gpu = [frame.to(device) for frame in decoded_frames]
            decoded_video = torch.stack(decoded_frames_gpu, dim=1)  # [B, T, C, H, W]
            del decoded_frames, decoded_frames_gpu  # 释放列表
        else:
            decoded_video = torch.stack(decoded_frames, dim=1)  # [B, T, C, H, W]
            del decoded_frames  # 释放列表
        
        return decoded_video
    
    def _warp_features(self, features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        使用光流扭曲特征
        
        Args:
            features (torch.Tensor): 特征 [B, C, H, W]
            flow (torch.Tensor): 光流 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 扭曲后的特征
        """
        B, C, H, W = features.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=features.device),
            torch.arange(W, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 应用光流
        flow_grid = grid + flow
        
        # 归一化到[-1, 1]
        flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
        flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # 重排列为grid_sample格式
        flow_grid = flow_grid.permute(0, 2, 3, 1)
        
        # 双线性插值
        warped_features = F.grid_sample(
            features, flow_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped_features
    
    def _decode_swin_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        使用Swin Transformer解码特征
        
        Args:
            features (torch.Tensor): 输入特征 [B, hidden_dim, H, W]
            
        Returns:
            torch.Tensor: 解码的帧 [B, out_channels, H', W']
        """
        B, C, H, W = features.shape
        
        # 特征重塑
        x = self.feature_reshape(features)
        
        # 重塑为序列格式
        x = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 通过Swin Transformer层（包含上采样）
        # 修复OOM：在每层之间清理GPU缓存，减少内存碎片
        for i, layer in enumerate(self.swin_layers):
            x = layer(x, use_checkpoint=self.training)  # 确保使用梯度检查点
        
        # 重塑为特征图格式
        # 经过 PatchReverseMerging 上采样后，特征图尺寸应为原始图像的尺寸
        # 例如：从 56×56 -> 112×112 -> 224×224
        num_patches = x.shape[1]
        
        # 直接使用 img_size 来确定输出尺寸（简化逻辑，去除复杂的推断代码）
        # 经过3层Swin层上采样后，分辨率应为：patches_resolution × 4 = img_size
        # 第1层：56×56 -> 56×56（不上采样）
        # 第2层：56×56 -> 112×112（上采样2倍）
        # 第3层：112×112 -> 224×224（上采样2倍）
        # 总上采样倍数：2 × 2 = 4倍
        h, w = self.img_size[0], self.img_size[1]
        
        # 断言：确保 num_patches == h * w（验证上采样逻辑的正确性）
        # 如果断言失败，说明上采样配置有误，应立即报错而不是试图"修复"
        assert num_patches == h * w, (
            f"VideoJSCCDecoder上采样维度不匹配："
            f"num_patches={num_patches}, img_size={self.img_size}, h*w={h*w}。"
            f"经过上采样后，num_patches应该等于img_size的乘积。"
            f"请检查上采样配置：patches_resolution={self.patches_resolution}, "
            f"预期最终分辨率应为 {self.patches_resolution[0]*4}×{self.patches_resolution[1]*4}。"
        )
        
        x = x.view(B, h, w, -1).permute(0, 3, 1, 2)  # [B, hidden_dim, H', W']
        
        # 输出投影（通道映射 + Sigmoid）
        x = self.output_proj(x)
        
        return x
    
    def _apply_semantic_guidance(
        self, 
        video_features: torch.Tensor, 
        semantic_context: torch.Tensor
    ) -> torch.Tensor:
        """
        【修复】应用语义引导的交叉注意力 - 使用标准CrossAttention模块
        
        Args:
            video_features (torch.Tensor): 视频特征 [B, C, H, W]
            semantic_context (torch.Tensor): 语义上下文 [B, seq_len, D_text]
            
        Returns:
            torch.Tensor: 增强后的视频特征
        """
        B, C, H, W = video_features.shape
        
        # 将视频特征重塑为序列格式
        video_seq = video_features.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 维度对齐
        # 注意：semantic_aligner 已在 __init__ 中定义，不应在 forward 中动态创建
        # 如果维度不匹配，使用预定义的语义对齐层
        if video_seq.shape[-1] != semantic_context.shape[-1]:
            # 验证维度是否与预定义的对齐层匹配
            if (semantic_context.shape[-1] != self.semantic_context_dim or 
                video_seq.shape[-1] != self.hidden_dim):
                raise RuntimeError(
                    f"语义对齐层维度不匹配："
                    f"semantic_context维度={semantic_context.shape[-1]}, 预期={self.semantic_context_dim}; "
                    f"video_seq维度={video_seq.shape[-1]}, 预期={self.hidden_dim}。"
                    f"请检查VideoJSCCDecoder的初始化参数。"
                )
            aligned_semantic = self.semantic_aligner(semantic_context)
        else:
            aligned_semantic = semantic_context
        
        if aligned_semantic.shape[0] != B:
            raise RuntimeError(
                f"semantic_context batch mismatch: semantic_batch={aligned_semantic.shape[0]}, expected={B}. "
                "请检查 DataLoader/Collate 的 batch 对齐。"
            )
        
        # 【修复】使用标准CrossAttention模块（与ImageJSCCDecoder保持一致）
        # CrossAttention 内部会处理 Query 和 Key/Value 序列长度不同的情况
        # 视频特征作为Query，语义上下文作为Key和Value
        enhanced_video_seq, attn_weights = self.cross_attention(
            query=video_seq,  # [B, H*W, hidden_dim]
            guide_vector=aligned_semantic,  # [B, seq_len, hidden_dim]
            return_attention=True
        )
        if attn_weights is not None:
            self.last_semantic_gate_stats["mean"] = float(attn_weights.mean().item())
            self.last_semantic_gate_stats["std"] = float(attn_weights.std().item())
        
        # 残差连接
        enhanced_video_seq = video_seq + enhanced_video_seq
        
        # 重塑回视频格式
        enhanced_video_features = enhanced_video_seq.transpose(1, 2).view(B, C, H, W)
        
        return enhanced_video_features
    
    def reset_hidden_state(self):
        """重置隐藏状态"""
        self.hidden_state = None
