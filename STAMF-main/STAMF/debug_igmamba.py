import torch
import torch.nn as nn
import os
import sys

# 确保能导入 Models 包
sys.path.append(os.getcwd())

# 引入你的模型 (请确保 Models/USOD_Net.py 中的类名也是 ImageDepthNet，或者在这里修改为 IGMambaNet)
try:
    from Models.USOD_Net import DualStreamIGMambaNet as Network
    print("[Success] 成功导入 DualStreamIGMambaNet")
except ImportError:
    try:
        from Models.USOD_Net import IGMambaNet as Network
        print("[Success] 成功导入 IGMambaNet")
    except ImportError as e:
        print(f"[Error] 无法导入模型类: {e}")
        exit()

def test_igmamba_flow():
    print("\n=== 开始 IGMamba 模型冒烟测试 ===")
    
    # 1. 模拟配置参数 (Args)
    class MockArgs:
        def __init__(self):
            self.img_size = 224
            # 如果你的 T2t_vit 需要 pretrained 路径，这里设为 None 或 False 来跳过加载权重
            self.pretrained_model = None 
            # 如果有其他必要参数，请在这里添加，例如:
            self.tokens_type = 'transformer' 
            
    args = MockArgs()
    
    # 2. 实例化网络
    print("\n[Step 1] 初始化网络...")
    try:
        # 注意：如果在 USOD_Net 中有 pretrained=True，确保它能处理 None 路径或者手动设为 False
        # 这里为了测试结构，建议暂时修改 USOD_Net 或传入参数使其不加载真实权重
        model = Network(args)
        model.cuda()
        model.train() # 开启训练模式以测试 Dropout 等
        print(" -> 网络初始化成功!")
    except Exception as e:
        print(f" -> [Error] 网络初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 创建 Dummy 数据 (模拟 DataLoader)
    print("\n[Step 2] 生成假数据 (Batch Size=2)...")
    batch_size = 2
    # 模拟 RGB 图像 (B, 3, 224, 224)
    dummy_images = torch.randn(batch_size, 3, 224, 224).cuda()
    # 模拟 Depth 图像 (B, 3, 224, 224) (假设 Backbone 处理 3 通道)
    dummy_depths = torch.randn(batch_size, 3, 224, 224).cuda()
    # 模拟 GT Mask (B, 1, 224, 224)，用于梯度 Loss 验证
    dummy_labels = torch.rand(batch_size, 1, 224, 224).cuda()
    
    # 4. 前向传播测试
    print("\n[Step 3] 执行前向传播 (Forward)...")
    try:
        # DualStream 需要两个输入
        outputs = model(dummy_images, dummy_depths)
        print(" -> 前向传播完成!")
    except Exception as e:
        print(f" -> [Error] 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 验证输出结构
    print("\n[Step 4] 验证输出结构...")
    # 我们期望输出是一个 tuple: (rgb_saliency_list, depth_saliency_list, rgb_grad_pred, depth_grad_pred)
    if isinstance(outputs, tuple) and len(outputs) == 4:
        rgb_saliency_list, depth_saliency_list, rgb_grad_pred, depth_grad_pred = outputs
        print(" -> 输出格式正确: (RGB Sal, Depth Sal, RGB Grad, Depth Grad)")
        
        # 检查显著性图
        if isinstance(rgb_saliency_list, (list, tuple)):
            print(f"    - RGB 显著性图数量: {len(rgb_saliency_list)}")
            print(f"    - RGB d1 尺寸: {rgb_saliency_list[0].shape}")
        
        if isinstance(depth_saliency_list, (list, tuple)):
            print(f"    - Depth (Fused) 显著性图数量: {len(depth_saliency_list)}")
            print(f"    - Depth d1 尺寸: {depth_saliency_list[0].shape}")
            
        # 检查梯度图
        print(f"    - RGB 梯度图尺寸: {rgb_grad_pred.shape}")
        print(f"    - Depth 梯度图尺寸: {depth_grad_pred.shape}")
            
    else:
        print(f" -> [Error] 输出格式不符合预期! 收到类型: {type(outputs)}")
        if isinstance(outputs, (list, tuple)):
            print(f"    长度: {len(outputs)}")
        return

    # 6. 模拟 Loss 计算和反向传播
    print("\n[Step 5] 模拟 Loss 和 反向传播 (Backward)...")
    try:
        # 简化的 loss 计算
        loss_sal = rgb_saliency_list[0].mean() + depth_saliency_list[0].mean()
        loss_grad = rgb_grad_pred.mean() + depth_grad_pred.mean()
        total_loss = loss_sal + loss_grad
        
        total_loss.backward()
        print(" -> 反向传播成功! 梯度流正常。")
    except Exception as e:
        print(f" -> [Error] 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== [PASS] 恭喜！LIQAM 集成验证通过 ===")


if __name__ == "__main__":
    test_igmamba_flow()