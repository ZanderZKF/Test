import torch
import os
import argparse
# Mock imports before importing Models
import timm.models.helpers
import timm.models
# Patching
dummy_load = lambda model, checkpoint_path, use_ema=False: None
timm.models.helpers.load_checkpoint = dummy_load
timm.models.load_checkpoint = dummy_load
try:
    import timm.models._helpers
    timm.models._helpers.load_checkpoint = dummy_load
except ImportError:
    pass

import sys
from unittest.mock import MagicMock
# sys.modules['timm.models.load_checkpoint'] = MagicMock() # This is tricky

from Models.USOD_Net import DualStreamIGMambaNet
from Training import get_gradient_map

# Mock Args
class Args:
    def __init__(self):
        self.Training = True
        self.Testing = False
        self.Evaluation = False
        self.data_root = '/root/autodl-tmp/'
        self.train_steps = 10
        self.img_size = 224
        self.pretrained_model = '' # Empty to avoid loading
        self.lr = 1e-4
        self.epochs = 1
        self.batch_size = 2
        self.save_model_dir = 'checkpoint/verify/'
        self.resume = None
        self.lr_decay_gamma = 0.1
        self.stepvalue1 = 5
        self.stepvalue2 = 8
        self.trainset = 'USOD10KK/TR'
        self.optimizer = 'adam'
        self.scheduler = 'step'
        self.use_experts_highfreq = True
        self.use_grad_ssim_loss = True
        self.gpu_id = '0'

def verify():
    args = Args()
    print("Initializing Net...")
    # Mock load_checkpoint to avoid file not found
    original_load = torch.load
    def mock_load(f, map_location=None):
        return {}
    torch.load = mock_load
    
    net = DualStreamIGMambaNet(args)
    net.cuda()
    net.train()
    
    # Restore torch.load
    torch.load = original_load
    
    print("Net Initialized. Running dummy forward/backward...")
    
    # Dummy Data
    B = args.batch_size
    images = torch.randn(B, 3, 224, 224).cuda()
    depths = torch.randn(B, 3, 224, 224).cuda() # Dataset converts depth to RGB (3 channels)
    
    # Labels
    label_224 = torch.rand(B, 1, 224, 224).cuda()
    
    # Forward
    outputs = net(images, depths)
    rgb_saliency_list, depth_saliency_list, rgb_grad_pred, depth_grad_pred = outputs
    
    print("Forward Pass Successful.")
    print(f"RGB Saliency List Length: {len(rgb_saliency_list)}")
    print(f"Gradient Pred Type: {type(rgb_grad_pred)}")
    if isinstance(rgb_grad_pred, list):
        print(f"Gradient List Length: {len(rgb_grad_pred)}")
        # Verify sizes
        for i, g in enumerate(rgb_grad_pred):
            print(f"  Grad {i} shape: {g.shape}")
    
    # Loss Calculation (simplified from Training.py)
    gt_gradient = get_gradient_map(label_224)
    
    # Just check if we can compute loss
    loss = torch.tensor(0.0).cuda()
    
    # Mock loss computation
    if isinstance(rgb_grad_pred, list):
        for p in rgb_grad_pred:
            if p.shape[-1] != 224:
                p = torch.nn.functional.interpolate(p, size=(224, 224), mode='bilinear', align_corners=False)
            loss += torch.mean((p - gt_gradient)**2)
    else:
        loss += torch.mean((rgb_grad_pred - gt_gradient)**2)
        
    print(f"Loss Computed: {loss.item()}")
    
    # Backward
    loss.backward()
    print("Backward Pass Successful.")
    
    print("Verification Passed!")

if __name__ == "__main__":
    verify()
