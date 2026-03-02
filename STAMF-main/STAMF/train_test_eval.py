import os
import torch
import argparse
import Training  # 确保 Training.py 已修改为支持双流
import Testing   # 确保 Testing.py 已修改为上述代码

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- 训练参数 ---
    parser.add_argument('--Training', default=False, type=str2bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33112', type=str)
    parser.add_argument('--data_root', default='/root/autodl-tmp/', type=str, help='data path')
    parser.add_argument('--train_steps', default=40000, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    # 注意：双流网络 RGB 分支会加载 pretrained，Depth 分支不加载，逻辑已在 USOD_Net 内部处理
    parser.add_argument('--pretrained_model', default='/root/autodl-tmp/STAMF-main/pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load Pretrained model')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate') # lr 5e-5
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--batch_size', default=12, type=int, help='batch size') # bs 10
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str)
    parser.add_argument('--resume', type=str, default=None)
    
    # --- 学习率衰减参数 ---
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float)
    parser.add_argument('--stepvalue1', default=30000, type=int)
    parser.add_argument('--stepvalue2', default=20000, type=int)
    parser.add_argument('--trainset', default='USOD10KK/TR', type=str)
    
    # --- 优化器与调度器参数 ---
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adam', 'adamw'], help='Optimizer choice')
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['step', 'cosine'], help='Scheduler choice')

    # --- 测试与评估参数 ---
    parser.add_argument('--Testing', default=True, type=str2bool, help='Testing or not')
    parser.add_argument('--Evaluation', default=True, type=str2bool, help='Evaluation or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='path to save saliency maps')
    # 确保这里的 test_paths 格式符合 evaluator 要求 (DatasetName/Set)
    parser.add_argument('--test_paths', type=str, default='USOD10KK/TE') 
    parser.add_argument('--save_dir', type=str, default='Evaluation/Result', help='path to save evaluation results')
    parser.add_argument('--use_experts_highfreq', default=True, type=str2bool, help='Enable Heterogeneous Experts in High-Frequency Branch')
    parser.add_argument('--use_grad_ssim_loss', default=True, type=str2bool, help='Enable SSIM loss for Gradient Branch')
    
    # --- Ablation & Configuration ---
    parser.add_argument('--depth_pretrained_path', type=str, default=None, help='Path to depth backbone pretrained weights')
    parser.add_argument('--low_pass_enabled', default=True, type=str2bool, help='Enable Low-Pass Filter for Mamba (TinyViM strategy)')
    parser.add_argument('--fusion_type', default='aff', type=str, choices=['linear', 'aff'], help='Fusion type in IGM (linear or aff)')

    args = parser.parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # 根据实际情况调整
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs.")

    # 1. 训练阶段
    if args.Training:
        print("\n>>> Start Training Phase")
        Training.train_net(num_gpus=num_gpus, args=args)
        # 训练完成后强制开启测试
        args.Testing = True
    
    # 2. 测试与评估阶段
    if args.Testing:
        print("\n>>> Start Testing & Evaluation Phase")
        Testing.start_testing_and_evaluation(args)
