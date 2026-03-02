
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Patch timm.models.load_checkpoint to avoid file not found errors during verification
try:
    import timm.models
    def mock_load_checkpoint(*args, **kwargs):
        print("Mock load_checkpoint called - skipping actual weight load for verification")
        return
    timm.models.load_checkpoint = mock_load_checkpoint
except ImportError:
    print("timm not found or failed to import")

# Add current directory to path
sys.path.append(os.getcwd())

from Models.USOD_Net import DualStreamIGMambaNet

class MockArgs:
    def __init__(self):
        self.use_experts_highfreq = True
        self.use_grad_ssim_loss = True
        self.pretrained_model = None # Skip loading inside model init if possible, but patched anyway
        # Add other potential args if needed by T2t_vit or others

def verify_training_step():
    print("=== Starting Training Feasibility Verification ===")
    
    # 1. Initialize Model
    print("Initializing Model...")
    args = MockArgs()
    try:
        model = DualStreamIGMambaNet(args).cuda()
        print("Model initialized successfully.")
    except Exception as e:
        print(f"FAILED to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Setup Optimizer
    print("Setting up Optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 3. Create Dummy Data
    print("Creating Dummy Data...")
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224).cuda()
    depth = torch.randn(batch_size, 1, 224, 224).cuda()
    gt_mask = torch.randint(0, 2, (batch_size, 1, 224, 224)).float().cuda()
    gt_grad = torch.randn(batch_size, 1, 224, 224).cuda() # Dummy gradient GT
    
    # 4. Forward Pass
    print("Running Forward Pass...")
    try:
        # rgb_saliency_list, depth_saliency_list, rgb_grad_pred, depth_grad_pred
        rgb_sal_list, depth_sal_list, rgb_grad_pred, depth_grad_pred = model(rgb, depth)
        print("Forward pass successful.")
    except Exception as e:
        print(f"FAILED during Forward Pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Calculate Loss
    print("Calculating Loss...")
    try:
        # Simple dummy loss for verification
        loss = 0
        for pred in rgb_sal_list:
            pred = torch.sigmoid(pred)
            pred = torch.nn.functional.interpolate(pred, size=gt_mask.shape[2:], mode='bilinear', align_corners=False)
            loss += torch.nn.functional.binary_cross_entropy(pred, gt_mask)
        
        for pred in depth_sal_list:
            pred = torch.sigmoid(pred)
            pred = torch.nn.functional.interpolate(pred, size=gt_mask.shape[2:], mode='bilinear', align_corners=False)
            loss += torch.nn.functional.binary_cross_entropy(pred, gt_mask)
            
        # Gradient loss
        rgb_grad_pred = torch.nn.functional.interpolate(rgb_grad_pred, size=gt_grad.shape[2:], mode='bilinear', align_corners=False)
        loss += torch.nn.functional.mse_loss(rgb_grad_pred, gt_grad)
        
        print(f"Loss calculated: {loss.item()}")
    except Exception as e:
        print(f"FAILED during Loss Calculation: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Backward Pass
    print("Running Backward Pass...")
    try:
        optimizer.zero_grad()
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"FAILED during Backward Pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Optimizer Step
    print("Running Optimizer Step...")
    try:
        optimizer.step()
        print("Optimizer step successful.")
    except Exception as e:
        print(f"FAILED during Optimizer Step: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=== Verification Successful: Training loop is feasible! ===")

if __name__ == "__main__":
    verify_training_step()
