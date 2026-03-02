import argparse
from Evaluation.main import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 预测图根目录
    parser.add_argument('--save_test_path_root', type=str,
                        default='preds/')

    # 测试集路径（相对 data_root）
    parser.add_argument('--test_paths', type=str,
                        default='USOD10KK/TE')

    # data_root，即 GT 的上级目录
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/')

    # 方法名（即 preds/USOD10KK/ 下的文件夹名称）
    parser.add_argument('--methods', type=str,
                        default='USODUVST_half_half4+USODUVST_half2+USODUVST_final')

    # 输出评测结果保存位置
    parser.add_argument('--save_dir', type=str,
                        default='Evaluation/Result')

    args = parser.parse_args()

    evaluate(args)
