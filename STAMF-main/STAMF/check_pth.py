import torch
import sys

path = '/root/autodl-tmp/STAMF-main/STAMF/checkpoint/IGMamba_final_20260128.pth'
try:
    state_dict = torch.load(path, map_location='cpu')
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    
    has_dqfm = any('DQFM' in k for k in keys)
    has_liqam = any('LIQAM' in k for k in keys)
    
    print(f"Contains DQFM keys: {has_dqfm}")
    print(f"Contains LIQAM keys: {has_liqam}")
    
    print("Sample keys:")
    for k in keys[:10]:
        print(k)
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
