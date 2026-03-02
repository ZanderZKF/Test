import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models # 这一行在原代码中被注释掉了，这里也保持原样

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # resnet = models.resnet34(pretrained=True) 

        ## -------------Decoder--------------
        # stage 5d
        # IGMamba Single Stream: Input is Bottleneck (384), changed from 768 (384+384)
        self.conv5d_1 = nn.Conv2d(384, 384, 3, padding=1) 
        self.bn5d_1 = nn.BatchNorm2d(384)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(384, 128, 3, dilation=2, padding=2)
        self.bn5d_m = nn.BatchNorm2d(128)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(128, 128, 3, dilation=2, padding=2)
        self.bn5d_2 = nn.BatchNorm2d(128)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(192, 128, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(128)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4d_m = nn.BatchNorm2d(128)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(128)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(192, 128, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(128)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3d_m = nn.BatchNorm2d(128)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d
        self.conv2d_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(64)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2d_m = nn.BatchNorm2d(64)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        # 注意：这里的输入通道是 67 (64来自上一层 + 3来自skip1)
        self.conv1d_1 = nn.Conv2d(67, 64, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 16, 3, padding=1)
        self.bn1d_m = nn.BatchNorm2d(16)
        self.relu1d_m = nn.ReLU(inplace=True)

        # 注意：这一层输出了 3 个通道，这就是 hd1 的通道数
        self.conv1d_2 = nn.Conv2d(16, 3, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(3)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upscore8  = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=True)
        self.upscore4  = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=True)
        self.upscore2  = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)

        ## -------------Side Output (Saliency Heads)--------------
        self.outconvb = nn.Conv2d(384, 1, 3, padding=1) # 768 -> 384
        self.outconv5 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(3, 1, 3, padding=1) # 输入是 hd1 (3通道)

        ## -------------【新增：梯度预测头】--------------
        # 在最高分辨率特征 hd1 上并行预测梯度图
        # 输入通道 3 (来自hd1)，输出通道 1 (梯度图)
        self.outconv_grad = nn.Conv2d(3, 1, 3, padding=1)
        
        # 多尺度梯度监督 (Hierarchical Perception)
        self.outconv_grad2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv_grad3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv_grad4 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv_grad5 = nn.Conv2d(128, 1, 3, padding=1)


    def forward(self, x_bottleneck, skip3, skip2, skip1):
        # x_bottleneck: [B, 384, 14, 14]
        
        # 5d
        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(x_bottleneck)))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 14 -> 28

        # 4d
        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, skip3), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 28 -> 56

        # 3d
        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1((torch.cat((hx, skip2), 1)))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 56 -> 112

        # 2d
        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(hx)))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 112 -> 224

        # 1d
        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, skip1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        # hd1 是最终的高分辨率特征图，形状 [B, 3, 224, 224]
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output (Saliency Predictions)-------------
        db = self.outconvb(x_bottleneck)
        udb = self.upscore16(db)  # 14->224

        d5 = self.outconv5(hd5)
        ud5 = self.upscore16(d5)  # 14->224

        d4 = self.outconv4(hd4)
        ud4 = self.upscore8(d4)   # 28->224

        d3 = self.outconv3(hd3)
        ud3 = self.upscore4(d3)   # 56->224

        d2 = self.outconv2(hd2)
        ud2 = self.upscore2(d2)   # 112->224

        d1 = self.outconv1(hd1)   # 224->224

        ## -------------【新增：梯度预测】-------------
        # 利用最终特征 hd1 预测梯度图
        pred_gradient = self.outconv_grad(hd1) # [B, 1, 224, 224]
        
        # 多尺度梯度 (Hierarchical Perception)
        grad2 = self.outconv_grad2(hd2)
        ugrad2 = self.upscore2(grad2)
        
        grad3 = self.outconv_grad3(hd3)
        ugrad3 = self.upscore4(grad3)
        
        grad4 = self.outconv_grad4(hd4)
        ugrad4 = self.upscore8(grad4)
        
        grad5 = self.outconv_grad5(hd5)
        ugrad5 = self.upscore16(grad5)

        # 将所有显著性输出打包成列表
        saliency_outputs = [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), 
                            F.sigmoid(d5), F.sigmoid(db), F.sigmoid(ud2), F.sigmoid(ud3), 
                            F.sigmoid(ud4), F.sigmoid(ud5), F.sigmoid(udb)]

        # 返回：(显著性列表, 经过Sigmoid激活的梯度预测图列表)
        # 使用 Sigmoid 是因为 GT 梯度图通常归一化到 [0,1]
        grad_outputs = [F.sigmoid(pred_gradient), F.sigmoid(ugrad2), F.sigmoid(ugrad3), F.sigmoid(ugrad4), F.sigmoid(ugrad5)]
        return saliency_outputs, grad_outputs