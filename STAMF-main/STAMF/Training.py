import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from dataset import get_loader
import math
import os
import pytorch_iou
import pytorch_ssim
# from Testing import start_testing_and_evaluation  # Moved inside main to avoid scope/circular import issues

# 【修改点 1】: 引入新的双流网络类
# 确保 Models/USOD_Net.py 中已经定义了 DualStreamIGMambaNet
# Use DualStreamIGMambaNet which contains LIQAM
from Models.USOD_Net import DualStreamIGMambaNet as NetModel
print("=> [Info] Using DualStreamIGMambaNet (with LIQAM) for training.")
# try:
#     # Ablation Study: Use AblationNet instead of DualStreamIGMambaNet
#     # from Models.USOD_Net import DualStreamIGMambaNet as NetModel
#     from Models.USOD_Net_Ablation import AblationNet as NetModel
#     print("=> [Info] Using AblationNet (Simple Fusion) for training.")
# except ImportError:
#     from Models.USOD_Net import ImageDepthNet as NetModel
#     print("Warning: AblationNet not found, falling back to ImageDepthNet.")

# ================= Loss Functions =================
criterion = nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=7, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
mse_loss = nn.MSELoss() 

def bce_ssim_loss(pred, target):
    bce_out = criterion(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    loss = bce_out + ssim_out
    return loss

def bce_iou_loss(pred, target):
    bce_out = criterion(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def get_gradient_map(x):
    # Scharr operator to get GT gradient (Aligned with High-Frequency Branch)
    # x is expected to be [B, 1, H, W]
    if x.size(1) == 3:
        x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
    kernel_x = torch.tensor([[3., 0., -3.], [10., 0., -10.], [3., 0., -3.]]).view(1, 1, 3, 3).to(x.device)
    kernel_y = torch.tensor([[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]]).view(1, 1, 3, 3).to(x.device)
    
    padding = nn.ReflectionPad2d(1)
    x_pad = padding(x.float())
    grad_x = F.conv2d(x_pad, kernel_x)
    grad_y = F.conv2d(x_pad, kernel_y)
    
    gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    min_v, _ = torch.min(gradient.view(gradient.size(0), -1), dim=1)
    max_v, _ = torch.max(gradient.view(gradient.size(0), -1), dim=1)
    min_v = min_v.view(gradient.size(0), 1, 1, 1)
    max_v = max_v.view(gradient.size(0), 1, 1, 1)
    gradient = (gradient - min_v) / (max_v - min_v + 1e-8)
    return gradient

# ================= Helper: 封装单分支的 Loss 计算 =================
def calc_saliency_loss(outputs_list, label_224, label_112, label_56, label_28, label_14, contour_224):
    """
    计算单个 Decoder 输出列表的综合损失 (Edge + BCE + IoU + Dice)
    """
    d1, d2, d3, d4, d5, db, ud2, ud3, ud4, ud5, udb = outputs_list

    # 1. BCE Loss
    bce_loss1 = criterion(d1, label_224)
    bce_loss2 = criterion(d2, label_112)
    bce_loss3 = criterion(d3, label_56)
    bce_loss4 = criterion(d4, label_28)
    bce_loss5 = criterion(d5, label_14)
    bce_loss6 = criterion(db, label_14)
    bce_total = bce_loss1 + bce_loss2 + bce_loss3 + bce_loss4 + bce_loss5 + bce_loss6

    # 2. IoU Loss
    iou_loss1 = bce_iou_loss(d1,  label_224)
    iou_loss2 = bce_iou_loss(ud2, label_224)
    iou_loss3 = bce_iou_loss(ud3, label_224)
    iou_loss4 = bce_iou_loss(ud4, label_224)
    iou_loss5 = bce_iou_loss(ud5, label_224)
    iou_loss6 = bce_iou_loss(udb, label_224)
    iou_total = iou_loss1 + iou_loss2 + iou_loss3 + iou_loss4 + iou_loss5 + iou_loss6

    # 3. Edge/SSIM Loss
    c_loss1 = bce_ssim_loss(d1,  contour_224)
    c_loss2 = bce_ssim_loss(ud2, label_224)
    c_loss3 = bce_ssim_loss(ud3, label_224)
    c_loss4 = bce_ssim_loss(ud4, label_224)
    c_loss5 = bce_ssim_loss(ud5, label_224)
    c_loss6 = bce_ssim_loss(udb, label_224)
    edge_total = c_loss1 + c_loss2 + c_loss3 + c_loss4 + c_loss5 + c_loss6

    # 4. Dice Loss
    d_loss1 = dice_loss(d1,   label_224)
    d_loss2 = dice_loss(ud2,  label_224)
    d_loss3 = dice_loss(ud3,  label_224)
    d_loss4 = dice_loss(ud4,  label_224)
    d_loss5 = dice_loss(ud5,  label_224)
    d_loss6 = dice_loss(udb,  label_224)
    dice_total = d_loss1 + d_loss2 + d_loss3 + d_loss4 + d_loss5 + d_loss6

    total_branch_loss = bce_total + iou_total + edge_total + dice_total
    
    return total_branch_loss, (bce_total, iou_total, edge_total, dice_total)

# ================= Utils =================
def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer

def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    fh.write(f'until_{epoch}_run_iter_num{whole_iter_num}\n')
    fh.write(f'{epoch}_epoch_total_loss{epoch_total_loss}\n')
    fh.write(f'{epoch}_epoch_loss{epoch_loss}\n\n')
    fh.close()

def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    with open(save_dir, 'a') as fh:
        fh.write(f'lr: {update_lr_group["lr"]}\n\n')

def train_net(num_gpus, args):
    mp.spawn(main, nprocs=1, args=(1, args))

# ================= Main Training Loop =================
def main(local_rank, num_gpus, args):
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)
    
    # 实例化双流网络
    net = NetModel(args)
    
    # Resume Logic (保持原样)
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"=> [Resume] Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            try:
                # 注意：如果网络结构变了，resume 可能会失败，建议从头训练或仅加载 backbone
                net.load_state_dict(new_state_dict, strict=False) 
                print(f"=> [Resume] Successfully loaded.")
            except Exception as e:
                print(f"=> [Resume Error] {e}")
        else:
            print(f"=> [Resume Warning] No checkpoint found.")
    else:
        # 如果没有 resume，确保加载 T2T-ViT 的预训练权重 (Backbone)
        # 这通常在 Models/USOD_Net.py 中处理，但这里可以打印确认
        print("=> [Info] No resume checkpoint provided. Backbone should be initialized from T2T-ViT weights.")

    net.train()
    net.cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    # 优化器设置
    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    if hasattr(args, 'optimizer') and args.optimizer == 'adam':
        print("=> [Info] Using Adam optimizer (User Requested).")
        optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                                {'params': other_params, 'lr': args.lr}],
                               weight_decay=0) # Adam usually doesn't need high weight decay or handled differently
    else:
        print("=> [Info] Using AdamW optimizer.")
        optimizer = optim.AdamW([{'params': base_params, 'lr': args.lr * 0.1},
                                 {'params': other_params, 'lr': args.lr}],
                                weight_decay=1e-4)
    
    # Scheduler Setup
    if hasattr(args, 'scheduler') and args.scheduler == 'step':
        print(f"=> [Info] Using StepLR Scheduler. Decay at {args.stepvalue1} and {args.stepvalue2}.")
        scheduler = None # Will handle manually in loop
    else:
        print("=> [Info] Using CosineAnnealingLR Scheduler.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_gpus, rank=local_rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                               pin_memory=True, sampler=sampler, drop_last=True)

    print(f'Starting training: Steps: {args.train_steps}, Batch: {args.batch_size}, LR: {args.lr}')
    
    N_train = len(train_loader) * args.batch_size
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    writer = SummaryWriter('runs/dual_stream_logs')

    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch + 1}/{args.epochs}, LR: {args.lr}')
        epoch_total_loss = 0
        epoch_loss_rec = 0 # 仅记录 BCE 用于显示

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            # 【修改点 2】: 获取双流输入
            # depths 变量不再注释，需要传入 GPU
            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            images = Variable(images.cuda(local_rank, non_blocking=True))
            depths = Variable(depths.cuda(local_rank, non_blocking=True)) # 启用 Depth 输入
            label_224 = Variable(label_224.cuda(local_rank, non_blocking=True))
            contour_224 = Variable(contour_224.cuda(local_rank, non_blocking=True))
            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()), \
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            # 【修改点 3】: 前向传播接收 4 个输出
            # rgb_list: RGB Decoder 的显著性输出列表
            # depth_list: Depth Decoder 的显著性输出列表
            # rgb_grad: RGB 分支的梯度预测
            # depth_grad: Depth 分支的梯度预测
            net_outputs = net(images, depths)
            rgb_saliency_list, depth_saliency_list, rgb_grad_pred, depth_grad_pred = net_outputs

            # --- 准备梯度监督 GT ---
            gt_gradient = get_gradient_map(label_224)
            g_obj = gt_gradient * label_224 # Object-masked gradient

            # Helper for Gradient Loss (Multi-scale)
            def calc_grad_list_loss(grad_list, target):
                loss = 0
                for pred in grad_list:
                    if pred.shape[-1] != target.shape[-1]:
                        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
                    loss += mse_loss(pred, target)
                    if hasattr(args, 'use_grad_ssim_loss') and args.use_grad_ssim_loss:
                        loss += (1 - ssim_loss(pred, target))
                return loss / len(grad_list)

            # --- 计算 RGB 分支 Loss ---
            loss_rgb_sal, (rgb_bce, rgb_iou, rgb_edge, rgb_dice) = calc_saliency_loss(
                rgb_saliency_list, label_224, label_112, label_56, label_28, label_14, contour_224
            )
            
            # Multi-scale Gradient Loss
            if isinstance(rgb_grad_pred, list):
                loss_rgb_grad = calc_grad_list_loss(rgb_grad_pred, g_obj)
            else:
                # Fallback for single tensor (old compatibility)
                if rgb_grad_pred.shape[-1] != 224:
                    rgb_grad_pred = F.interpolate(rgb_grad_pred, size=(224, 224), mode='bilinear', align_corners=False)
                loss_rgb_grad = mse_loss(rgb_grad_pred, g_obj)
                if hasattr(args, 'use_grad_ssim_loss') and args.use_grad_ssim_loss:
                    loss_rgb_grad += (1 - ssim_loss(rgb_grad_pred, g_obj))
            
            # --- 计算 Depth 分支 Loss ---
            loss_depth_sal, (depth_bce, depth_iou, depth_edge, depth_dice) = calc_saliency_loss(
                depth_saliency_list, label_224, label_112, label_56, label_28, label_14, contour_224
            )
            
            if isinstance(depth_grad_pred, list):
                loss_depth_grad = calc_grad_list_loss(depth_grad_pred, g_obj)
            else:
                if depth_grad_pred.shape[-1] != 224:
                    depth_grad_pred = F.interpolate(depth_grad_pred, size=(224, 224), mode='bilinear', align_corners=False)
                loss_depth_grad = mse_loss(depth_grad_pred, g_obj)
                if hasattr(args, 'use_grad_ssim_loss') and args.use_grad_ssim_loss:
                    loss_depth_grad += (1 - ssim_loss(depth_grad_pred, g_obj))

            # --- Total Loss ---
            # 策略: 两路同时监督，权重 1:1，梯度 Loss 权重 0.5 (可调)
            lambda_grad = 0.5
            total_loss = (loss_rgb_sal + lambda_grad * loss_rgb_grad) + \
                         (loss_depth_sal + lambda_grad * loss_depth_grad)

            # --- Logging ---
            step = epoch * len(train_loader) + i
            writer.add_scalar('Loss/Total', total_loss.item(), step)
            writer.add_scalar('Loss/RGB_Sal', loss_rgb_sal.item(), step)
            writer.add_scalar('Loss/Depth_Sal', loss_depth_sal.item(), step)
            
            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss_rec += rgb_bce.cpu().data.item() # 只记录一个作为参考

            if i % 20 == 0:
                print('Iter: {} | Total: {:.4f} | RGB_Sal: {:.4f} | Depth_Sal: {:.4f}'.format(
                    whole_iter_num, total_loss.item(), loss_rgb_sal.item(), loss_depth_sal.item()
                ))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            whole_iter_num += 1

            # Save Logic (Enhanced with timestamp and immediate evaluation)
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            
            # 为了快速验证，我们使用传入的 args.train_steps 来控制保存点
            # 这里的逻辑是：在 1/4, 1/2, 和结束时保存
            # 注意：如果 train_steps 很小（比如 100），这些条件也能触发
            
            save_condition_1 = (whole_iter_num == args.train_steps // 4) and (whole_iter_num > 0)
            save_condition_2 = (whole_iter_num == args.train_steps // 2) and (whole_iter_num > 0)
            save_condition_final = (whole_iter_num == args.train_steps) and (whole_iter_num > 0)

            if save_condition_1:
                save_name = f'IGMamba_1_4_{current_date}.pth'
                save_path = os.path.join(args.save_model_dir, save_name)
                torch.save(net.state_dict(), save_path)
                print(f"Model saved to {save_path}. Starting evaluation...")
                from Testing import start_testing_and_evaluation
                start_testing_and_evaluation(args, current_model_path=save_path)

            elif save_condition_2:
                save_name = f'IGMamba_half_{current_date}.pth'
                save_path = os.path.join(args.save_model_dir, save_name)
                torch.save(net.state_dict(), save_path)
                print(f"Model saved to {save_path}. Starting evaluation...")
                from Testing import start_testing_and_evaluation
                start_testing_and_evaluation(args, current_model_path=save_path)

            elif save_condition_final:
                save_name = f'IGMamba_final_{current_date}.pth'
                save_path = os.path.join(args.save_model_dir, save_name)
                torch.save(net.state_dict(), save_path)
                print(f"Model saved to {save_path}. Starting evaluation...")
                from Testing import start_testing_and_evaluation
                start_testing_and_evaluation(args, current_model_path=save_path)
                return 0


            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                if scheduler is None: # Manual StepLR
                    optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                    save_lr('./loss.txt', optimizer)
                else:
                    pass 

        # Update Scheduler at the end of epoch
        if scheduler is not None:
            scheduler.step()
            print(f'Epoch {epoch+1} finished. Avg Loss: {epoch_total_loss / iter_num}. Current LR: {scheduler.get_last_lr()}')
        else:
            # Manual Step LR printing
            print(f'Epoch {epoch+1} finished. Avg Loss: {epoch_total_loss / iter_num}. Current LR: {optimizer.param_groups[1]["lr"]}')
            
        save_loss('./loss.txt', whole_iter_num, epoch_total_loss / iter_num, epoch_loss_rec / iter_num, epoch + 1)
    
    # Final Safety Save (Ensure model is saved even if loops finish without hitting exact train_steps)
    print("=> [Info] Training loop finished. Saving final model...")
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_name = f'IGMamba_final_end_{current_date}.pth'
    save_path = os.path.join(args.save_model_dir, save_name)
    torch.save(net.state_dict(), save_path)
    print(f"Final Model saved to {save_path}. Starting evaluation...")
    from Testing import start_testing_and_evaluation
    start_testing_and_evaluation(args, current_model_path=save_path)
    return 0
