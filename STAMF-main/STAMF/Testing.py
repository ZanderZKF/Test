import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
# 【修改点 1】导入新的双流网络
try:
    from Models.USOD_Net import DualStreamIGMambaNet as NetModel
except ImportError:
    from Models.USOD_Net import ImageDepthNet as NetModel
    print("Warning: DualStreamIGMambaNet not found, falling back to ImageDepthNet.")

from torch.utils import data
import numpy as np
import os
import cv2
import glob 
from collections import OrderedDict

# 导入评估模块
from Evaluation.main import evaluate as eval_all 

def run_test_loop(args, checkpoints):
    """
    遍历所有 checkpoint 并调用 test_net 进行预测。
    """
    all_method_names = []
    
    # 确保保存路径存在
    if not os.path.exists(args.save_test_path_root):
        os.makedirs(args.save_test_path_root)

    for model_path in checkpoints:
        print('\n' + '=' * 80)
        print(f'Processing checkpoint: {model_path}')
        print('=' * 80)

        # 使用 checkpoint 文件名作为方法名 (例如 IGMamba_final)
        ckpt_name = os.path.splitext(os.path.basename(model_path))[0]
        method_name = ckpt_name  
        all_method_names.append(method_name)

        # 设置当前模型路径并测试
        args.model_path = model_path
        test_net(args, method_name)
    
    # -------------------------
    # 所有预测生成完毕，开始统一评估
    # -------------------------
    all_method_names = list(dict.fromkeys(all_method_names)) # 去重
    args.methods = '+'.join(all_method_names) # 拼接方法名传给 evaluator

    if not hasattr(args, 'save_dir'):
        args.save_dir = 'Evaluation/Result'
    os.makedirs(args.save_dir, exist_ok=True)

    print('\nStarting Evaluation for methods:', args.methods)
    eval_all(args) # 调用 Evaluation/main.py

    print('\nAll processes (Training -> Testing -> Evaluation) finished!')


def test_net(args, method_name):
    cudnn.benchmark = True

    # 【修改点 2】实例化双流网络
    net = NetModel(args)
    net.cuda()
    net.eval()
    
    model_path = args.model_path
    print(f'Loading model from {model_path}')
    
    # 加载权重 (处理 DDP 的 module. 前缀)
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    test_paths = args.test_paths.split('+')

    for test_dir_img in test_paths:
        # 获取测试数据加载器
        test_dataset = get_loader(
            test_dir_img,
            args.data_root,
            args.img_size,
            mode='test' # mode='test' 会返回 (image, depth, w, h, path)
        )

        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        dataset_name = test_dir_img.split('/')[0]
        print(f'Testing on dataset: {dataset_name}, Size: {len(test_loader.dataset)}')

        time_list = []
        with torch.no_grad():
            for i, data_batch in enumerate(test_loader):
                # 【修改点 3】获取双流数据
                images, depths, image_w, image_h, image_path = data_batch
                
                images = Variable(images.cuda())
                depths = Variable(depths.cuda()) # 传入 Depth

                starts = time.time()
                
                # 【修改点 4】前向传播 (传入双流)
                # 返回: (rgb_list, depth_list, rgb_grad, depth_grad)
                outputs = net(images, depths)
                
                # 我们取 Depth Stream 的结果，因为它是被 RGB 引导增强过的
                # outputs[1] 是 depth_saliency_list
                # list[0] 是 d1 (最高分辨率输出)
                depth_saliency_list = outputs[1]
                pred_map = depth_saliency_list[0]
                
                ends = time.time()
                time_list.append(ends - starts)

                # 后处理
                image_w, image_h = int(image_w[0]), int(image_h[0])
                
                # 插值回原始尺寸
                pred_map = F.interpolate(pred_map, size=(image_h, image_w), mode='bilinear', align_corners=False)
                pred_map = pred_map.sigmoid().data.cpu().numpy().squeeze()
                
                # 归一化
                pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
                
                # 保存路径构建
                filename = os.path.splitext(os.path.basename(image_path[0]))[0]
                
                # 保存结构: args.save_test_path_root / dataset_name / method_name / filename.png
                # 例如: preds/USOD10KK/IGMamba_final/123.png
                save_test_path = os.path.join(args.save_test_path_root, dataset_name, method_name)
                
                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path, exist_ok=True)

                cv2.imwrite(os.path.join(save_test_path, filename + '.png'), pred_map * 255)
            
            print(f'Finished {method_name} on {dataset_name}. Avg Time: {np.mean(time_list)*1000:.2f} ms')
            
        torch.cuda.empty_cache()


def start_testing_and_evaluation(args, current_model_path=None):
    """
    主入口函数。
    :param args: 包含配置参数的对象
    :param current_model_path: 如果提供，则只评估该模型（用于训练过程中的在线评估）
    """
    if current_model_path:
        # 单模型模式
        checkpoints = [current_model_path]
    else:
        # 批量模式：扫描目录
        checkpoint_dir = os.path.join(os.getcwd(), args.save_model_dir)
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'IGMamba_*.pth'))
        if not checkpoints:
            print(f"Warning: No checkpoints found in {checkpoint_dir}")
            return
        checkpoints.sort()

    if checkpoints:
        print(f"Starting evaluation for {len(checkpoints)} checkpoint(s)...")
        run_test_loop(args, checkpoints)

