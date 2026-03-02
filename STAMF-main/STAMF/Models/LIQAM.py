import torch
import torch.nn as nn
import torch.nn.functional as F

class IGHA(nn.Module):
    def __init__(self, dim):
        super(IGHA, self).__init__()
        # Illumination Projection
        # Old Version: Input channel = 1 (Raw Illumination Map)
        self.proj_ill = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Spatial Context Aggregation (Psi)
        # Old Version: Single layer convolution
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Output Projection
        self.f_out = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_target, f_source, ill_map):
        # Cross-Modal Consensus
        M_c = f_target * f_source
        
        # Illumination Projection
        # Resize ill_map to feature size
        curr_ill = F.interpolate(ill_map, size=f_target.shape[2:], mode='bilinear', align_corners=True)
        G_ill = self.proj_ill(curr_ill)
        
        # Spatial Aggregation
        S_gate = self.psi(M_c + G_ill)
        
        # Recalibration Map
        beta = self.f_out(M_c + S_gate + G_ill)
        
        return beta

class HFA(nn.Module):
    def __init__(self, dim):
        super(HFA, self).__init__()
        self.dim = dim
        
        # Multi-scale transform layers (simulated by pooling)
        # We process original scale (s1), 1/2 scale (s2), 1/4 scale (s3)
        # Here we just define MLP to process concatenated scores
        # Input to MLP: 3 scales * 3 pairs = 9 scores? 
        # Or as per paper: alignments for (r,d), (r,g), (d,g) across scales.
        
        # Let's assume we compute alignment for (rgb, depth) and (rgb, grad) and (depth, grad)
        # Total 3 pairs * 3 scales = 9 inputs if we concat all?
        # Based on code logic:
        # We predict alpha_d and alpha_g.
        
        # Simple MLP to fuse alignment scores
        self.mlp = nn.Sequential(
            nn.Linear(9, 32), # 3 pairs * 3 scales
            nn.ReLU(inplace=True),
            nn.Linear(32, 2 * dim), # Output 2*dim weights (alpha_d, alpha_g)
            nn.Sigmoid()
        )
        
        # Transform layers to align channels if needed (usually 1x1 conv)
        self.trans_r = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim), nn.ReLU())
        self.trans_d = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim), nn.ReLU())
        self.trans_g = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim), nn.ReLU())

    def _compute_alignment(self, f1, f2):
        # f1, f2: [B, C, H, W]
        # Alignment = GAP(f1 * f2) / (GAP(f1 + f2) + eps)
        inter = (f1 * f2).mean(dim=[2, 3]) # [B, C]
        union = (f1 + f2).mean(dim=[2, 3]) # [B, C]
        align = inter / (union + 1e-6)
        return align # [B, C]

    def forward(self, f_r, f_d, f_g):
        B, C, H, W = f_r.shape
        
        # Transform
        f_r = self.trans_r(f_r)
        f_d = self.trans_d(f_d)
        f_g = self.trans_g(f_g)
        
        # Multi-scale pooling
        scales = [1, 0.5, 0.25]
        align_scores = []
        
        pairs = [(f_r, f_d), (f_r, f_g), (f_d, f_g)]
        
        for s in scales:
            if s != 1:
                curr_h, curr_w = int(H*s), int(W*s)
                fr_s = F.adaptive_avg_pool2d(f_r, (curr_h, curr_w))
                fd_s = F.adaptive_avg_pool2d(f_d, (curr_h, curr_w))
                fg_s = F.adaptive_avg_pool2d(f_g, (curr_h, curr_w))
            else:
                fr_s, fd_s, fg_s = f_r, f_d, f_g
                
            # Compute alignments for this scale
            # We need a single scalar per pair per scale to feed into MLP? 
            # Or channel-wise? The paper says "channel-wise weights".
            # If MLP outputs C weights, inputs should be related to C.
            # But MLP input dim is fixed (9). This implies global alignment scores.
            
            # Re-reading: "concatenate them and pass through MLP to generate channel-wise weights"
            # If MLP is Linear(9, ...), inputs must be scalars (GAP over channel too?)
            # Let's assume we take mean over channels for the input to MLP
            
            a_rd = self._compute_alignment(fr_s, fd_s).mean(dim=1, keepdim=True) # [B, 1]
            a_rg = self._compute_alignment(fr_s, fg_s).mean(dim=1, keepdim=True)
            a_dg = self._compute_alignment(fd_s, fg_s).mean(dim=1, keepdim=True)
            
            align_scores.extend([a_rd, a_rg, a_dg])
            
        # Concat: [B, 9]
        mlp_in = torch.cat(align_scores, dim=1)
        
        # MLP -> [B, 2*C]
        weights = self.mlp(mlp_in)
        
        alpha_d = weights[:, :self.dim].unsqueeze(2).unsqueeze(3) # [B, C, 1, 1]
        alpha_g = weights[:, self.dim:].unsqueeze(2).unsqueeze(3)
        
        return alpha_d, alpha_g

class LIQAM(nn.Module):
    def __init__(self, dim):
        super(LIQAM, self).__init__()
        
        self.hfa = HFA(dim)
        self.igha_d = IGHA(dim)
        self.igha_g = IGHA(dim)
        
    def forward(self, f_rgb, f_depth, f_grad, ill_map):
        # 1. HFA: Channel-wise weights
        alpha_d, alpha_g = self.hfa(f_rgb, f_depth, f_grad)
        
        # 2. IGHA: Spatial weights
        beta_d = self.igha_d(f_rgb, f_depth, ill_map)
        beta_g = self.igha_g(f_rgb, f_grad, ill_map)
        
        # 3. Modulated Fusion
        # F_fused = Fr + (alpha_d * Fd * beta_d) + (alpha_g * Fg * beta_g)
        f_fused = f_rgb + (alpha_d * f_depth * beta_d) + (alpha_g * f_grad * beta_g)
        
        return f_fused
