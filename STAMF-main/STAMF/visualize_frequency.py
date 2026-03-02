import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from Models.USOD_Net import DualStreamIGMambaNet as NetModel
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MockArgs:
    def __init__(self):
        self.img_size = 224
        self.pretrained_model = '/root/autodl-tmp/STAMF-main/pretrained_model/80.7_T2T_ViT_t_14.pth.tar'
        # 添加其他可能需要的参数，根据 USOD_Net 的 __init__ 要求
        # 通常主要就是 img_size 和 pretrained_model

def load_model(model_path):
    args = MockArgs()
    model = NetModel(args).to(device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        
        # 处理 DDP 的 module. 前缀
        state_dict = torch.load(model_path, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: Model path {model_path} not found. Using random initialization (visualization might be noise).")
    model.eval()
    return model

def preprocess_image(image_path, depth_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    depth = Image.open(depth_path).convert('RGB')
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    depth_tensor = transform(depth).unsqueeze(0).to(device)
    
    return img_tensor, depth_tensor, np.array(img.resize((224, 224)))

# Hook 存储容器
features = {}

def get_features(name):
    def hook(model, input, output):
        # IGMambaModule_Symmetric 的 forward 返回 f_out, f_illum_flow, f_grad_flow
        # output 是一个 tuple: (f_out, f_illum_flow, f_grad_flow)
        if isinstance(output, tuple) and len(output) >= 3:
            features[name + '_illum'] = output[1].detach()
            features[name + '_grad'] = output[2].detach()
    return hook

def visualize_feature_map(feature_tensor, title, save_path):
    # feature_tensor: [B, C, H, W]
    # 1. 沿通道取平均 (B, 1, H, W)
    heatmap = torch.mean(feature_tensor, dim=1).squeeze()
    
    # 2. 归一化到 [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = heatmap.cpu().numpy()
    
    # 3. 转换为伪彩色图 (Jet colormap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 4. 保存
    cv2.imwrite(save_path, heatmap)
    print(f"Saved {title} to {save_path}")
    return heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/STAMF-main/STAMF/checkpoint/IGMamba_final.pth') # 指向您的权重
    parser.add_argument('--image_path', type=str, required=True, help='Path to RGB image')
    parser.add_argument('--depth_path', type=str, required=True, help='Path to Depth image')
    parser.add_argument('--output_dir', type=str, default='vis_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    model = load_model(args.model_path)

    # 2. 注册 Hook
    # 我们选择 Layer 3 (IGM3) 进行可视化，因为它通常包含较丰富的高级语义
    # 在 USOD_Net.py 中，self.IGM3 是第三层的 IGMambaModule_Symmetric
    target_layer = model.IGM3
    target_layer.register_forward_hook(get_features('IGM3'))

    # 3. 准备数据
    img_tensor, depth_tensor, original_img = preprocess_image(args.image_path, args.depth_path)

    # 4. 前向传播
    with torch.no_grad():
        model(img_tensor, depth_tensor)

    # 5. 可视化
    if 'IGM3_illum' in features and 'IGM3_grad' in features:
        # 保存原图
        cv2.imwrite(os.path.join(args.output_dir, 'original.jpg'), cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        
        # 可视化 Illumination Flow (Low Frequency)
        visualize_feature_map(
            features['IGM3_illum'], 
            'Illumination Flow', 
            os.path.join(args.output_dir, 'vis_illum_flow.jpg')
        )
        
        # 可视化 Gradient Flow (High Frequency)
        visualize_feature_map(
            features['IGM3_grad'], 
            'Gradient Flow', 
            os.path.join(args.output_dir, 'vis_grad_flow.jpg')
        )
        
        print("\nVisualization Complete!")
        print(f"Illumination Flow (Low Freq) saved to {os.path.join(args.output_dir, 'vis_illum_flow.jpg')}")
        print(f"Gradient Flow (High Freq) saved to {os.path.join(args.output_dir, 'vis_grad_flow.jpg')}")
        print("Expected Result: Illumination Flow should look smooth/global; Gradient Flow should look sharp/edge-like.")
    else:
        print("Error: Could not capture features. Check layer name or hook registration.")

if __name__ == '__main__':
    main()
