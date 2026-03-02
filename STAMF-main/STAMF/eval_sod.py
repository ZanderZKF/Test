import os
import cv2
import numpy as np


def get_files(dir_path):
    return sorted(
        [f for f in os.listdir(dir_path)
         if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
    )


# ---------- MAE ----------
def mae(pred, gt):
    # pred, gt: [0,1]
    return np.mean(np.abs(pred - gt))


# ---------- F-measure (max Fβ, β^2 = 0.3) ----------
def f_measure(pred, gt, beta2=0.3 * 0.3, eps=1e-8):
    # pred, gt: [0,1]
    gt_bin = gt > 0.5

    prec_list = []
    rec_list = []

    for t in np.linspace(0, 1, 256):
        p_bin = pred >= t
        tp = np.logical_and(p_bin, gt_bin).sum()
        fp = np.logical_and(p_bin, ~gt_bin).sum()
        fn = np.logical_and(~p_bin, gt_bin).sum()

        if tp + fp == 0 or tp + fn == 0:
            continue

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        prec_list.append(prec)
        rec_list.append(rec)

    if len(prec_list) == 0:
        return 0.0

    prec = np.array(prec_list)
    rec = np.array(rec_list)
    f = (1 + beta2) * prec * rec / (beta2 * prec + rec + eps)
    return f.max()


# ---------- E-measure (Enhanced-alignment, max over thresholds) ----------
def _e_measure_single(bin_pred, bin_gt, eps=1e-8):
    """
    bin_pred, bin_gt: binary maps {0,1}
    返回单一阈值下的 E-score
    """
    # image-level mean
    mu_pred = bin_pred.mean()
    mu_gt = bin_gt.mean()

    # bias matrices
    align_pred = bin_pred - mu_pred
    align_gt = bin_gt - mu_gt

    # alignment matrix ξ
    numerator = 2 * align_pred * align_gt
    denominator = align_pred ** 2 + align_gt ** 2 + eps
    xi = numerator / denominator

    # enhanced alignment Φ
    phi = (xi + 1) ** 2 / 4

    return phi.mean()


def e_measure(pred, gt, eps=1e-8):
    """
    pred, gt: [0,1]
    按论文做法，对预测图按 256 个阈值二值化，取 E-score 最大值
    """
    gt_bin = gt > 0.5
    if gt_bin.size == 0:
        return 0.0

    scores = []
    for t in np.linspace(0, 1, 256):
        bin_pred = pred >= t
        scores.append(_e_measure_single(bin_pred.astype(np.float32),
                                        gt_bin.astype(np.float32),
                                        eps=eps))
    return np.max(scores)


# ---------- S-measure (Structure-measure, Sα) ----------
def _object_similarity(pred_region, gt_region, eps=1e-8):
    """
    object-aware 部分 So 里用到的前景/背景相似度
    这里 pred_region 是前景像素或背景像素的 saliency 值
    gt_region 只用来算权重（area 比例）
    """
    if pred_region.size == 0:
        return 0.0

    mu = pred_region.mean()
    # 按论文的形式：2μ/(μ^2 + 1)
    return (2 * mu) / (mu * mu + 1 + eps)


def _ssim(pred_region, gt_region, eps=1e-8):
    """
    region-aware 部分 Sr 里用到的 SSIM
    """
    if pred_region.size == 0:
        return 0.0

    mu_x = pred_region.mean()
    mu_y = gt_region.mean()
    sigma_x2 = ((pred_region - mu_x) ** 2).mean()
    sigma_y2 = ((gt_region - mu_y) ** 2).mean()
    sigma_xy = ((pred_region - mu_x) * (gt_region - mu_y)).mean()

    # 论文里常用的 SSIM 常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return num / (den + eps)


def s_measure(pred, gt, alpha=0.5, eps=1e-8):
    """
    pred, gt: [0,1]
    按 S-measure 定义，返回 Sα
    """
    # 将 GT 变成 {0,1}
    gt_bin = gt > 0.5

    h, w = gt_bin.shape
    area = h * w

    # 特殊情况：没有前景 或 全前景
    fg_num = gt_bin.sum()
    if fg_num == 0:
        # 全背景：结构性越“黑”越好
        return 1.0 - pred.mean()
    if fg_num == area:
        # 全前景：结构性越“白”越好
        return pred.mean()

    # -------- So: object-aware 部分 --------
    fg_mask = gt_bin
    bg_mask = ~gt_bin

    w_fg = fg_mask.mean()
    w_bg = bg_mask.mean()

    pred_fg = pred[fg_mask]
    pred_bg = pred[bg_mask]

    # 前景 object 相似度
    o_fg = _object_similarity(pred_fg, gt_bin[fg_mask], eps=eps)

    # 背景用 (1 - S) 再做一次相似度
    o_bg = _object_similarity(1 - pred_bg, 1 - gt_bin[bg_mask], eps=eps)

    So = w_fg * o_fg + w_bg * o_bg

    # -------- Sr: region-aware 部分 --------
    # 使用 GT 的质心将图像分成四个区域
    ys, xs = np.where(gt_bin)
    y_cent = int(np.round(ys.mean()))
    x_cent = int(np.round(xs.mean()))

    # 防止越界
    y_cent = np.clip(y_cent, 1, h - 1)
    x_cent = np.clip(x_cent, 1, w - 1)

    regions = []
    # 左上
    regions.append((slice(0, y_cent), slice(0, x_cent)))
    # 右上
    regions.append((slice(0, y_cent), slice(x_cent, w)))
    # 左下
    regions.append((slice(y_cent, h), slice(0, x_cent)))
    # 右下
    regions.append((slice(y_cent, h), slice(x_cent, w)))

    Sr = 0.0
    for r in regions:
        pr = pred[r]
        gr = gt_bin[r]
        w_r = pr.size / float(area)
        Sr += w_r * _ssim(pr, gr, eps=eps)

    # -------- S-measure 综合 --------
    return alpha * So + (1 - alpha) * Sr


# ---------- 评估一个预测目录 ----------
def eval_one_dir(pred_dir, gt_dir):
    pred_files = get_files(pred_dir)

    mae_list = []
    fm_list = []
    em_list = []
    sm_list = []

    for name in pred_files:
        pred_path = os.path.join(pred_dir, name)
        gt_path = os.path.join(gt_dir, name)  # 假设 GT 文件名一样

        if not os.path.exists(gt_path):
            print("GT not found:", gt_path)
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            print("Read error:", pred_path, gt_path)
            continue

        # 归一化到 [0, 1]
        pred = pred.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0

        # resize 到 GT 大小
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        mae_list.append(mae(pred, gt))
        fm_list.append(f_measure(pred, gt))
        em_list.append(e_measure(pred, gt))
        sm_list.append(s_measure(pred, gt))

    return (np.mean(mae_list),
            np.mean(fm_list),
            np.mean(em_list),
            np.mean(sm_list))


if __name__ == "__main__":
    # 根据你自己的路径改这里
    data_root = "/root/autodl-tmp/"
    dataset = "USOD10KK/TR"      # 和测试时用的 test_paths 保持一致
    gt_dir = os.path.join(data_root, dataset, "GT")

    pred_root = "/root/autodl-tmp/STAMF-main/STAMF/preds"
    ckpt_dirs = [
        "UVST_half+half4.pth",
        "UVST_half2.pth"
    ]

    for ckpt in ckpt_dirs:
        pred_dir = os.path.join(
            pred_root,
            dataset.split('/')[0],   # USOD10KK
            "USOD",
            ckpt
        )

        print("Evaluate:", ckpt)
        print("Pred dir:", pred_dir)
        print("GT dir  :", gt_dir)

        mae_val, fm_val, em_val, sm_val = eval_one_dir(pred_dir, gt_dir)

        print("  MAE    : {:.4f}".format(mae_val))
        print("  F_max  : {:.4f}".format(fm_val))
        print("  E_max  : {:.4f}".format(em_val))
        print("  S_alpha: {:.4f}".format(sm_val))
        print("------------------------------------------")
