import argparse
import os
import Testing
from Testing import start_testing_and_evaluation

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
    
    # 基本参数
    parser.add_argument('--data_root', default='/root/autodl-tmp/', type=str, help='data path')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--pretrained_model', default=None, type=str) # 测试时不需要加载 backbone 预训练
    
    # 测试参数
    parser.add_argument('--save_test_path_root', default='preds_manual/', type=str, help='path to save saliency maps')
    parser.add_argument('--test_paths', type=str, default='USOD10KK/TE') 
    parser.add_argument('--save_dir', type=str, default='Evaluation/Result_Manual', help='path to save evaluation results')
    
    # 目标权重
    parser.add_argument('--target_checkpoint', type=str, required=True, help='Path to the checkpoint to evaluate')

    args = parser.parse_args()

    print(f"Starting manual evaluation for: {args.target_checkpoint}")
    
    # 调用 Testing.py 中的入口函数
    start_testing_and_evaluation(args, current_model_path=args.target_checkpoint)
