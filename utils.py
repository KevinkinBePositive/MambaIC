import torch
import torch.nn.functional as F
from torchvision import transforms
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None

def calc_psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y)
    max_val = float(max_val)
    max_val_tensor = torch.tensor(max_val, device=x.device)
    psnr = 20 * torch.log10(max_val_tensor) - 10 * torch.log10(mse)
    return psnr

def calc_ms_ssim(x, y, max_val=1.0):
    if ms_ssim is None:
        raise ImportError('Please install pytorch_msssim for MS-SSIM support.')
    # x, y: (B, C, H, W), 0~1
    return ms_ssim(x, y, data_range=max_val, size_average=True) 