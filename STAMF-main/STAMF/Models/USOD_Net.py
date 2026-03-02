import torch
import torch.nn as nn
# 假设这些模块都在当前目录或 Python 路径下
from .priors import PriorGenerator
from .t2t_vit import T2t_vit_t_14
from .DAM_module import IGMambaModule_Symmetric
from .Decoder_Dconv import Decoder
from .LIQAM import LIQAM

class DualStreamIGMambaNet(nn.Module):
    def __init__(self, args):
        super(DualStreamIGMambaNet, self).__init__()
        
        # 1. 物理先验生成器
        self.prior_generator = PriorGenerator()
        load_pretrained = True
        if hasattr(args, 'pretrained_model') and args.pretrained_model is None:
            load_pretrained = False
            print("[Info] args.pretrained_model is None, skipping backbone pretrained weights.")

        # 2. 双流骨干网络 (Encoders)
        # RGB Backbone
        self.rgb_backbone = T2t_vit_t_14(pretrained=load_pretrained, args=args)
        # Depth Backbone (结构相同，也可以设为不加载预训练以适应深度图)
        # 支持独立加载深度骨干权重
        depth_pretrained_path = getattr(args, 'depth_pretrained_path', None)
        if depth_pretrained_path:
            print(f"[Info] Loading Depth Backbone weights from {depth_pretrained_path}")
            self.depth_backbone = T2t_vit_t_14(pretrained=False, args=args)
            checkpoint = torch.load(depth_pretrained_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            msg = self.depth_backbone.load_state_dict(state_dict, strict=False)
            print(f"[Info] Depth Backbone loaded: {msg}")
        else:
            # Depth Backbone 同样使用 T2t_vit_t_14 结构，并加载预训练权重。
            # 虽然深度图与 RGB 域不同，但加载 ImageNet 预训练权重作为初始特征提取器
            # 通常比随机初始化收敛更快、效果更好（Transfer Learning）。
            # 网络会在训练初期快速调整第一层卷积以适应深度图分布。
            self.depth_backbone =  T2t_vit_t_14(pretrained=load_pretrained, args=args)
            
        # 3. 对称 IGM 模块 (仅用于 RGB 分支提取 Flows)
        # Reverted to original Symmetric IGM
        self.IGM1 = IGMambaModule_Symmetric(dim=3)   # Layer 1
        self.IGM2 = IGMambaModule_Symmetric(dim=64)  # Layer 2
        self.IGM3 = IGMambaModule_Symmetric(dim=64)  # Layer 3
        self.IGM_Bottleneck = IGMambaModule_Symmetric(dim=384) # Layer 4

        # 4. LIQAM 融合模块 (替换 DQFM)
        self.LIQAM1 = LIQAM(dim=3)
        self.LIQAM2 = LIQAM(dim=64)
        self.LIQAM3 = LIQAM(dim=64)
        self.LIQAM_Bottleneck = LIQAM(dim=384)

        # 5. 双解码器 (Decoders)
        self.rgb_decoder = Decoder()
        self.depth_decoder = Decoder()

    def _reshape_tokens(self, x):
        # 辅助函数: Tokens -> Image
        if x.dim() == 4: return x
        B, N, C = x.shape
        H = W = int(N**0.5)
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, rgb_input, depth_input):
        # --- Step 1: 先验生成 ---
        illumination, gradient_prior = self.prior_generator(rgb_input)

        # --- Step 2: RGB Stream (Encoder + IGM) ---
        r_bn, r_1_8, r_1_4, r_1_1, _, _ = self.rgb_backbone(rgb_input)
        # Reshape
        r_bn = self._reshape_tokens(r_bn)
        r_1_8 = self._reshape_tokens(r_1_8)
        r_1_4 = self._reshape_tokens(r_1_4)
        r_1_1 = self._reshape_tokens(r_1_1)

        # IGM 增强 & 提取 Flows (光照流/梯度流)
        # rX_enh: 增强后的RGB特征 (包含高低频)
        # rX_illum: 纯净的光照流 (低频)
        # rX_grad: 纯净的梯度流 (高频)
        r1_enh, r1_illum, r1_grad = self.IGM1(r_1_1, illumination, gradient_prior)
        r2_enh, r2_illum, r2_grad = self.IGM2(r_1_4, illumination, gradient_prior)
        r3_enh, r3_illum, r3_grad = self.IGM3(r_1_8, illumination, gradient_prior)
        rbn_enh, rbn_illum, rbn_grad = self.IGM_Bottleneck(r_bn, illumination, gradient_prior)

        # --- Step 3: Depth Stream (Encoder Only) ---
        d_bn, d_1_8, d_1_4, d_1_1, _, _ = self.depth_backbone(depth_input)
        d_bn = self._reshape_tokens(d_bn)
        d_1_8 = self._reshape_tokens(d_1_8)
        d_1_4 = self._reshape_tokens(d_1_4)
        d_1_1 = self._reshape_tokens(d_1_1)

        # --- Step 4: LIQAM 融合 (Key Logic) ---
        # F_fused = LIQAM(Fr, Fd, Fg, Illum)
        # 注意: 这里 Fr 我们使用增强后的 RGB 特征 (rX_enh), 
        # Fg 使用纯净的梯度流 (rX_grad) 用于对齐计算
        # 注意: LIQAM_Old 期望 ill_map 为单通道原始光照图，而非 IGM 输出的 Illum Flow
        
        d1_fused = self.LIQAM1(r1_enh, d_1_1, r1_grad, illumination)
        d2_fused = self.LIQAM2(r2_enh, d_1_4, r2_grad, illumination)
        d3_fused = self.LIQAM3(r3_enh, d_1_8, r3_grad, illumination)
        dbn_fused = self.LIQAM_Bottleneck(rbn_enh, d_bn, rbn_grad, illumination)

        # --- Step 5: Decoders ---
        # RGB Decoder (使用 RGB 增强特征)
        rgb_saliency_list, rgb_grad_pred = self.rgb_decoder(rbn_enh, r3_enh, r2_enh, r1_enh)
        
        # Depth Decoder (使用 LIQAM 融合后的特征)
        depth_saliency_list, depth_grad_pred = self.depth_decoder(dbn_fused, d3_fused, d2_fused, d1_fused)

        # 返回两组预测结果
        return rgb_saliency_list, depth_saliency_list, rgb_grad_pred, depth_grad_pred

# 兼容旧代码的别名
ImageDepthNet = DualStreamIGMambaNet
