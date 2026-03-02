import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# 引用统一的 Mamba 块构建函数
# 假设 transformer_mamba_block.py 提供了标准的实现 (基于 mamba_ssm 库)
# 确保该文件在同一目录下或者在 python path 中
from .transformer_mamba_block import create_block

# 核心创新模块: IGMambaModule (基于物理先验的频率分解与选择性全局建模)
# ==============================================================================
class IGMambaModule_Swapped(nn.Module):
    """
    消融实验专用：反转物理分支。
    - 将 梯度增强的高频特征 (F_high) 送入 Mamba (预期会丢失细节)
    - 将 光照引导的低频特征 (F_low) 作为残差 (预期无法进行全局校正)
    """
    def __init__(self, dim):
        super(IGMambaModule_Swapped, self).__init__()
        
        # 定义保持不变
        self.illum_conv = nn.Sequential(
            nn.Conv2d(dim + 1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Mamba 还是那个 Mamba
        self.mamba_layer = create_block(
            d_model=dim, 
            rms_norm=True,      
            layer_idx=0, 
            if_bimamba=True     
        )
        self.norm = nn.LayerNorm(dim) 

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, illumination, gradient):
        B, C, H, W = x.shape
        
        # 预处理保持不变
        curr_illum = F.interpolate(illumination, size=(H, W), mode='bilinear', align_corners=False)
        curr_grad = F.interpolate(gradient, size=(H, W), mode='bilinear', align_corners=False)

        # === 变量生成保持不变 ===
        # 1. 低频分量 F_low (代表光照)
        m_illum = self.illum_conv(torch.cat([x, curr_illum], dim=1))
        f_low = x * m_illum 

        # 2. 高频分量 F_high (代表梯度/边缘)
        f_high = x * (1 + self.alpha * curr_grad)

        # === 【关键修改：交换处理路径】 ===
        
        # 路径 A (Mamba路径): 这次我们把 [高频 F_high] 扔进去
        # 预期后果：Mamba 的长程积分效应会平滑掉 F_high 里的边缘细节
        x_mamba_in = f_high.flatten(2).transpose(1, 2)
        x_processed, _ = self.mamba_layer(x_mamba_in) 
        x_processed = self.norm(x_processed)
        f_mamba_out = x_processed.transpose(1, 2).reshape(B, C, H, W)

        # 路径 B (残差路径): 这次我们把 [低频 F_low] 放在这里
        # 预期后果：F_low 失去了全局传播机会，光照校正受限
        # f_out = Mamba输出 + gamma * 残差
        f_out = f_mamba_out + self.gamma * f_low

        return f_out

class IGMambaModule(nn.Module):
    """
    IGMamba 核心模块实现。
    
    设计思想：基于物理先验的频率分解与选择性全局建模。
    
    功能步骤：
    1. 接收输入特征图(x)、光照先验图(illumination)和梯度先验图(gradient)。
    2. 预处理：将先验图插值到与输入特征图相同的尺寸。
    3. 低频分解：利用光照先验图生成注意力掩码，提取受光照影响较大的低频特征(f_low)。
    4. 高频分解：利用梯度先验图增强输入特征中的高频边缘细节(f_high)。
    5. 全局建模：仅将平滑的低频特征(f_low)送入双向 Mamba 模块进行全局上下文建模，避免 Mamba 平滑掉高频细节。
    6. 混合重构：将 Mamba 处理后的全局特征与保留的高频残差特征进行加权融合。
    """
    def __init__(self, dim):
        super(IGMambaModule, self).__init__()
        
        # 1. 光照引导的低频融合层 (Illumination-Guided Gating)
        # 将特征与光照图拼接，通过卷积和 Sigmoid 生成光照掩码
        self.illum_conv = nn.Sequential(
            nn.Conv2d(dim + 1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 2. Mamba 全局传播模块 (Global Propagation)
        # 使用统一接口构建 Mamba 块用于处理低频特征
        # 启用双向扫描 (if_bimamba=True) 对视觉任务捕捉全局信息至关重要
        self.mamba_layer = create_block(
            d_model=dim, 
            rms_norm=True,      # 推荐使用 RMSNorm
            layer_idx=0, 
            if_bimamba=True     # 【关键】启用双向扫描
        )
        self.norm = nn.LayerNorm(dim) # Mamba 输出后的归一化层

        # 3. 梯度增强的高频模块参数 (Gradient Enhancement)
        # 可学习的标量权重，控制梯度信息增强的强度
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # 4. 最终融合层参数 (Hybrid Reconstruction)
        # 可学习的标量权重，控制高频残差注入的比例
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, illumination, gradient):
        B, C, H, W = x.shape
        
        # === 预处理：对齐先验图尺寸 ===
        # 将光照和梯度先验调整到当前特征图的分辨率
        curr_illum = F.interpolate(illumination, size=(H, W), mode='bilinear', align_corners=False)
        curr_grad = F.interpolate(gradient, size=(H, W), mode='bilinear', align_corners=False)

        # === Step 1: 分解 - 低频分量 (F_low) ===
        # 利用光照图生成掩码，强调光照区域，抑制散射严重的区域，提取低频基础特征
        m_illum = self.illum_conv(torch.cat([x, curr_illum], dim=1))
        f_low = x * m_illum 

        # === Step 2: 分解 - 高频分量 (F_high) ===
        # 利用梯度图显式增强特征中的高频边缘信息，这条路径将绕过 Mamba
        f_high = x * (1 + self.alpha * curr_grad)

        # === Step 3: Mamba 全局传播 (仅针对低频) ===
        # 维度变换: [B, C, H, W] -> [B, HW, C] 以适配 Mamba 的序列输入格式
        x_mamba_in = f_low.flatten(2).transpose(1, 2)
        
        # Mamba 前向传播 (捕捉长距离依赖)
        # 注意：create_block 返回的通常是 (output, residual)，这里我们只需要 output
        x_global, _ = self.mamba_layer(x_mamba_in) 
        x_global = self.norm(x_global)
        
        # 维度变换回: [B, HW, C] -> [B, C, H, W]
        f_global = x_global.transpose(1, 2).reshape(B, C, H, W)

        # === Step 4: 混合重构 (Hybrid Reconstruction) ===
        # 将全局处理后的低频特征与保留下来的高频残差特征进行融合
        f_out = f_global + self.gamma * f_high

        return f_out
        

# ==============================================================================
# 基础注意力模块 (被 DAM_module 使用)
# ==============================================================================
class CA_Enhance(nn.Module):
    """
    通道注意力增强模块 (Channel Attention Mechanism)。
    用于强调重要的特征通道。
    """
    def __init__(self, in_planes, ratio=16):
        super(CA_Enhance, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 使用两个 1x1 卷积构建 MLP，先降维再升维
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth):
        # 注意：这里的输入通常是主特征和跨层连接特征的拼接
        x = torch.cat((rgb, depth), dim=1)
        # 全局最大池化 -> MLP -> Sigmoid 生成通道注意力权重
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        attention_weights = self.sigmoid(max_out)
        # 将权重应用到需要增强的特征上 (这里选择增强 depth 输入)
        depth_out = depth.mul(attention_weights)
        return depth_out

class SA_Enhance(nn.Module):
    """
    空间注意力增强模块 (Spatial Attention Mechanism)。
    用于强调重要的空间区域。
    """
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 使用大卷积核 (7x7 或 3x3) 提取空间特征
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度进行最大池化，压缩为单通道空间图
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 卷积 -> Sigmoid 生成空间注意力权重
        x_out = self.conv1(max_out)
        attention_weights = self.sigmoid(x_out)
        return attention_weights


# ==============================================================================
# 双重注意力模块: DAM_module (统一实现)
# ==============================================================================
class DAM_module(nn.Module):
    """
    双重注意力模块 (Dual Attention Module, DAM)。
    
    结构：串联应用通道注意力 (CA) 和空间注意力 (SA)。
    用于在解码器的跳跃连接处增强特征融合效果。
    """
    def __init__(self, in_planes, ratio=16):
        super(DAM_module, self).__init__()
        # 实例化基础注意力子模块
        self.self_CA_Enhance = CA_Enhance(in_planes, ratio)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        # Step 1: 通道注意力增强
        # 利用 rgb 和 depth 特征计算通道权重，并增强 depth 特征
        x_d = self.self_CA_Enhance(rgb, depth)
        
        # Step 2: 空间注意力增强
        # 基于通道增强后的特征计算空间权重
        sa_weights = self.self_SA_Enhance(x_d)
        
        # 将空间权重应用到特征上
        depth_enhance = depth.mul(sa_weights)
        return depth_enhance
# ----------------------------------------------------------------
# 1. 局部 CNN 模块 (用于高频梯度流的精修)
# ----------------------------------------------------------------
class LocalCNNBlock(nn.Module):
    def __init__(self, dim):
        super(LocalCNNBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        return self.relu(out)

# ----------------------------------------------------------------
# 2. 对称 IGMamba 模块 (Symmetric IGM)
#    - 低频路径: Mamba (Global) -> Illumination Flow
#    - 高频路径: CNN (Local)   -> Gradient Flow
# ----------------------------------------------------------------

# Attention Feature Fusion (AFF) - Ported from AIMNet
class AFF(nn.Module):
    def __init__(self, channels, activation, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        if inter_channels < 1:
            inter_channels = 1

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, residual):
        # 自动对齐尺寸（裁剪右边多余的部分）
        min_h = min(x.size(2), residual.size(2))
        min_w = min(x.size(3), residual.size(3))
        x = x[:, :, :min_h, :min_w]
        residual = residual[:, :, :min_h, :min_w]

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class IGMambaModule_Symmetric(nn.Module):
    def __init__(self, dim):
        super(IGMambaModule_Symmetric, self).__init__()
        
        # --- Path 1: 低频光照流 (Mamba Global) ---
        self.illum_conv = nn.Sequential(
            nn.Conv2d(dim + 1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Low-Pass Filter: Decouple frequencies for Mamba (TinyViM style)
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # 使用 create_block 构建 Mamba 层，确保参数与你的 mamba_block 定义一致
        self.mamba_layer = create_block(d_model=dim, layer_idx=0, if_bimamba=True) 
        self.norm = nn.LayerNorm(dim)

        # --- Path 2: 高频梯度流 (CNN Local) ---
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.cnn_layer = LocalCNNBlock(dim)

        # --- 融合参数 ---
        # self.gamma = nn.Parameter(torch.tensor(0.1))
        self.aff = AFF(dim, nn.LeakyReLU(0.1, True))

    def forward(self, x, illumination, gradient):
        B, C, H, W = x.shape
        
        # 预处理先验图
        curr_illum = F.interpolate(illumination, size=(H, W), mode='bilinear', align_corners=False)
        curr_grad = F.interpolate(gradient, size=(H, W), mode='bilinear', align_corners=False)

        # === 1. 生成光照流 (Illumination Flow) ===
        m_illum = self.illum_conv(torch.cat([x, curr_illum], dim=1))
        f_low = x * m_illum
        
        # Explicit Low-Pass Filtering
        f_low_filtered = self.low_pass(f_low)
        
        x_mamba_in = f_low_filtered.flatten(2).transpose(1, 2)
        x_global, _ = self.mamba_layer(x_mamba_in) # Mamba 处理
        x_global = self.norm(x_global)
        f_illum_flow = x_global.transpose(1, 2).reshape(B, C, H, W)

        # === 2. 生成梯度流 (Gradient Flow) ===
        f_high_raw = x * (1 + self.alpha * curr_grad) # 梯度注入
        f_grad_flow = self.cnn_layer(f_high_raw)     # CNN 精修

        # === 3. 融合 ===
        # f_out = f_illum_flow + self.gamma * f_grad_flow
        f_out = self.aff(f_illum_flow, f_grad_flow)

        return f_out, f_illum_flow, f_grad_flow

class GradientDetailExperts(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        mid = dim
        self.expert0 = nn.Sequential(
            nn.Conv2d(dim, mid, 3, 1, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.expert1 = nn.Sequential(
            nn.Conv2d(dim, mid, 3, 1, 2, dilation=2, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 3, 1, 2, dilation=2, bias=False), nn.BatchNorm2d(dim)
        )
        hp = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
        self.lap = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        with torch.no_grad():
            w = hp.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
            self.lap.weight.copy_(w)
        
        # Local Gate: Pixel-wise decision based on feature + gradient prior
        self.proj = nn.Conv2d(dim + 1, 3, 1, bias=False)
        
        # Global Gate: Image-level decision based on global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        reduction_dim = max(1, dim // 4)
        self.global_proj = nn.Sequential(
            nn.Conv2d(dim, reduction_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_dim, 3, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, grad_prior):
        g = F.interpolate(grad_prior, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Local logits
        local_logits = self.proj(torch.cat([x, g], dim=1))
        
        # Global logits
        global_logits = self.global_proj(self.global_pool(x))
        
        # Combine logits
        gate = self.softmax(local_logits + global_logits)
        
        e0 = self.expert0(x)
        e1 = self.expert1(x)
        e2 = self.lap(x)
        w0 = gate[:, 0:1]
        w1 = gate[:, 1:2]
        w2 = gate[:, 2:3]
        out = w0 * e0 + w1 * e1 + w2 * e2
        return F.relu(out)

class IGMambaModule_Symmetric_Experts(nn.Module):
    def __init__(self, dim, low_pass_enabled=True, fusion_type='aff'):
        super().__init__()
        self.low_pass_enabled = low_pass_enabled
        self.fusion_type = fusion_type
        
        self.illum_conv = nn.Sequential(
            nn.Conv2d(dim + 1, 1, 1), nn.Sigmoid()
        )
        
        # Low-Pass Filter / Downsampling for Mamba (TinyViM Strategy)
        # TinyViM/FreqMamba research suggests Mamba works best with decoupled low-frequencies.
        # We use a stride of 2 to extract the "Coarse" low-frequency component for global modeling.
        self.down_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Use RepDW to process the high-frequency residue (local details) lost during downsampling
        self.local_conv = RepDW(dim)
        
        self.mamba_layer = create_block(d_model=dim, layer_idx=0, if_bimamba=True)
        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.experts = GradientDetailExperts(dim)
        
        if self.fusion_type == 'linear':
            self.gamma = nn.Parameter(torch.tensor(0.1))
        else:
            self.aff = AFF(dim, nn.LeakyReLU(0.1, True))

    def forward(self, x, illumination, gradient):
        B, C, H, W = x.shape
        curr_illum = F.interpolate(illumination, size=(H, W), mode='bilinear', align_corners=False)
        curr_grad = F.interpolate(gradient, size=(H, W), mode='bilinear', align_corners=False)
        
        # 1. Low Frequency Branch (Illumination)
        m_illum = self.illum_conv(torch.cat([x, curr_illum], dim=1))
        f_low = x * m_illum
        
        if self.low_pass_enabled:
            # --- TinyViM-inspired Frequency Decoupling Strategy ---
            # 1. Downsample to get Coarse/Low-Freq component
            f_low_down = self.down_pool(f_low)
            
            # 2. Compute Residual (High-Freq details lost in downsampling)
            f_low_coarse_up = F.interpolate(f_low_down, size=(H, W), mode='bilinear', align_corners=False)
            f_low_residue = f_low - f_low_coarse_up
            # Enhance local details in the residue with RepDW
            f_low_residue = self.local_conv(f_low_residue)
            
            # 3. Process Coarse component with Mamba (Global Context)
            x_mamba_in = f_low_down.flatten(2).transpose(1, 2)
            x_global, _ = self.mamba_layer(x_mamba_in)
            x_global = self.norm(x_global)
            
            # Reshape back (taking into account the downsampled size)
            H_down, W_down = f_low_down.shape[2], f_low_down.shape[3]
            f_mamba_out_down = x_global.transpose(1, 2).reshape(B, C, H_down, W_down)
            
            # 4. Upsample Mamba output and add Residual back
            f_mamba_out_up = F.interpolate(f_mamba_out_down, size=(H, W), mode='bilinear', align_corners=False)
            f_illum_flow = f_mamba_out_up + f_low_residue
        else:
            # Standard Mamba processing without frequency decoupling
            x_mamba_in = f_low.flatten(2).transpose(1, 2)
            x_global, _ = self.mamba_layer(x_mamba_in)
            x_global = self.norm(x_global)
            f_illum_flow = x_global.transpose(1, 2).reshape(B, C, H, W)
        
        # 2. High Frequency Branch (Gradient)
        f_high_raw = x * (1 + self.alpha * curr_grad)
        f_grad_flow = self.experts(f_high_raw, curr_grad)
        
        # 3. Fusion
        if self.fusion_type == 'linear':
            f_out = f_illum_flow + self.gamma * f_grad_flow
        else:
            f_out = self.aff(f_illum_flow, f_grad_flow)
        
        return f_out, f_illum_flow, f_grad_flow

# ----------------------------------------------------------------
# 3. DQFM 门控单元 (支持差异化输入)
# ----------------------------------------------------------------
class DQFM_Gate_Unit(nn.Module):
    def __init__(self, dim):
        super(DQFM_Gate_Unit, self).__init__()
        
        # 1. 确保中间通道数至少为 8 (修复之前的 RuntimeError)
        mid_dim = max(8, dim // 4)

        # 2. DHA Branch (空间门控)
        self.dha_conv = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            nn.Conv2d(mid_dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 3. DQW Branch (质量门控)
        self.dqw_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, depth_feat, illum_flow=None, grad_flow=None):
        B, C, H, W = depth_feat.shape
        
        dha_map = 1.0
        dqw_weight = 1.0

        # --- 逻辑 1: 空间门控 (DHA) ---
        if illum_flow is not None:
            dha_map = self.dha_conv(illum_flow * depth_feat)
        
        # --- 逻辑 2: 质量门控 (DQW) ---
        if grad_flow is not None:
            # 计算每通道的 IoU
            # inter/union shape: (B, C, 1)
            inter = torch.mean((grad_flow * depth_feat).flatten(2), dim=2, keepdim=True)
            union = torch.mean((grad_flow + depth_feat).flatten(2), dim=2, keepdim=True) + 1e-6
            iou = inter / union
            
            # 【关键修复】: 在通道维度取平均，得到全局 IoU
            # (B, C, 1) -> (B, 1)
            iou_global = torch.mean(iou, dim=1).view(B, 1) 
            
            # 输入 MLP 生成全局权重 (B, 1, 1, 1)
            dqw_weight = self.dqw_mlp(iou_global).view(B, 1, 1, 1)

        final_gate = dha_map * dqw_weight
        depth_gated = depth_feat * final_gate + depth_feat 
        
        return depth_gated, final_gate
