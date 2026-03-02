import torch
import torch.nn as nn

class Mamba(nn.Module):
    """
    非官方、简化版的“伪 Mamba”：
    - 接口兼容：至少支持 d_model、d_state、d_conv、expand 这些参数
    - 输入输出形状：(B, L, C) -> (B, L, C)
    - 用 depthwise conv + GELU + pointwise conv 模拟一个序列混合块
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        hidden_dim = int(d_model * expand)

        # depthwise 卷积：只在时间维度卷积，各通道独立
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=d_model,   # depthwise
        )

        # pointwise 卷积：通道混合
        self.pw_conv1 = nn.Conv1d(d_model, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv1d(hidden_dim, d_model, kernel_size=1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        """
        x: (B, L, C)
        """
        # 先做 LayerNorm
        x_norm = self.norm(x)        # (B, L, C)

        # 转成 (B, C, L) 做 1D 卷积
        x_t = x_norm.transpose(1, 2) # (B, C, L)

        y = self.dw_conv(x_t)
        y = self.pw_conv1(y)
        y = self.act(y)
        y = self.pw_conv2(y)

        # 转回 (B, L, C)
        y = y.transpose(1, 2)

        # 残差
        return x + y
