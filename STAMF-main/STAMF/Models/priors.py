import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorGenerator(nn.Module):
    def __init__(self):
        super(PriorGenerator, self).__init__()
        # 定义Sobel算子用于计算梯度
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def get_illumination(self, img):
        # 简化的Retinex光照分量提取：取RGB通道的最大值作为光照强度的近似
        # IGMamba 论文中提到光照图捕捉主要强度分布 [cite: 1452]
        illumination, _ = torch.max(img, dim=1, keepdim=True)
        return illumination

    def get_gradient(self, img):
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img

        kx = torch.tensor([[3., 0., -3.], [10., 0., -10.], [3., 0., -3.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray, kx, padding=1)
        grad_y = F.conv2d(gray, ky, padding=1)

        gradient = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)
        g_min = gradient.view(gradient.size(0), -1).min(dim=1)[0].view(gradient.size(0), 1, 1, 1)
        g_max = gradient.view(gradient.size(0), -1).max(dim=1)[0].view(gradient.size(0), 1, 1, 1)
        gradient = (gradient - g_min) / (g_max - g_min + 1e-8)
        return gradient

    def forward(self, x):
        return self.get_illumination(x), self.get_gradient(x)
