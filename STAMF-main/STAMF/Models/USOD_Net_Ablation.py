import torch
import torch.nn as nn
import torch.nn.functional as F
from .priors import PriorGenerator
from .t2t_vit import T2t_vit_t_14
from .DAM_module import IGMambaModule_Symmetric
from .Decoder_Dconv import Decoder

class SimpleFusion(nn.Module):
    def __init__(self, dim):
        super(SimpleFusion, self).__init__()
        # Input: RGB(dim) + Depth(dim) + Grad(dim) + Illum(dim)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Illumination projection to match dim if needed (usually ill_flow is already dim)
        # But to be safe, we assume inputs are already aligned in dim except ill_map might need check
        
    def forward(self, f_rgb, f_depth, f_grad, f_illum):
        # Resize illumination flow if needed (usually already same size)
        if f_illum.shape[2:] != f_rgb.shape[2:]:
            f_illum = F.interpolate(f_illum, size=f_rgb.shape[2:], mode='bilinear', align_corners=True)
            
        # Concatenate
        cat_feat = torch.cat([f_rgb, f_depth, f_grad, f_illum], dim=1)
        
        # Fuse
        out = self.fusion_conv(cat_feat)
        return out

class AblationNet(nn.Module):
    def __init__(self, args):
        super(AblationNet, self).__init__()
        
        # 1. 物理先验生成器
        self.prior_generator = PriorGenerator()
        load_pretrained = True
        if hasattr(args, 'pretrained_model') and args.pretrained_model is None:
            load_pretrained = False

        # 2. 双流骨干网络
        self.rgb_backbone = T2t_vit_t_14(pretrained=load_pretrained, args=args)
        self.depth_backbone =  T2t_vit_t_14(pretrained=load_pretrained, args=args)
        
        # 3. 对称 IGM 模块 (保留，只消融 LIQAM)
        self.IGM1 = IGMambaModule_Symmetric(dim=3)
        self.IGM2 = IGMambaModule_Symmetric(dim=64)
        self.IGM3 = IGMambaModule_Symmetric(dim=64)
        self.IGM_Bottleneck = IGMambaModule_Symmetric(dim=384)

        # 4. 简单的拼接融合 (替换 LIQAM)
        self.Fusion1 = SimpleFusion(dim=3)
        self.Fusion2 = SimpleFusion(dim=64)
        self.Fusion3 = SimpleFusion(dim=64)
        self.Fusion_Bottleneck = SimpleFusion(dim=384)

        # 5. 双解码器
        self.rgb_decoder = Decoder()
        self.depth_decoder = Decoder()

    def _reshape_tokens(self, x):
        if x.dim() == 4: return x
        B, N, C = x.shape
        H = W = int(N**0.5)
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, rgb_input, depth_input):
        illumination, gradient_prior = self.prior_generator(rgb_input)

        r_bn, r_1_8, r_1_4, r_1_1, _, _ = self.rgb_backbone(rgb_input)
        r_bn = self._reshape_tokens(r_bn)
        r_1_8 = self._reshape_tokens(r_1_8)
        r_1_4 = self._reshape_tokens(r_1_4)
        r_1_1 = self._reshape_tokens(r_1_1)

        r1_enh, r1_illum, r1_grad = self.IGM1(r_1_1, illumination, gradient_prior)
        r2_enh, r2_illum, r2_grad = self.IGM2(r_1_4, illumination, gradient_prior)
        r3_enh, r3_illum, r3_grad = self.IGM3(r_1_8, illumination, gradient_prior)
        rbn_enh, rbn_illum, rbn_grad = self.IGM_Bottleneck(r_bn, illumination, gradient_prior)

        d_bn, d_1_8, d_1_4, d_1_1, _, _ = self.depth_backbone(depth_input)
        d_bn = self._reshape_tokens(d_bn)
        d_1_8 = self._reshape_tokens(d_1_8)
        d_1_4 = self._reshape_tokens(d_1_4)
        d_1_1 = self._reshape_tokens(d_1_1)

        # --- Simple Fusion ---
        d1_fused = self.Fusion1(r1_enh, d_1_1, r1_grad, r1_illum)
        d2_fused = self.Fusion2(r2_enh, d_1_4, r2_grad, r2_illum)
        d3_fused = self.Fusion3(r3_enh, d_1_8, r3_grad, r3_illum)
        dbn_fused = self.Fusion_Bottleneck(rbn_enh, d_bn, rbn_grad, rbn_illum)

        rgb_saliency_list, rgb_grad_pred = self.rgb_decoder(rbn_enh, r3_enh, r2_enh, r1_enh)
        depth_saliency_list, depth_grad_pred = self.depth_decoder(dbn_fused, d3_fused, d2_fused, d1_fused)

        return rgb_saliency_list, depth_saliency_list, rgb_grad_pred, depth_grad_pred
