# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset import NightlightSRDataset



@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


    #Data
    tif_folder: str="."
    hr_size: int = 128
    scale: int = 2
    batch_size: int = 16
    num_workers: int = 2
    val_split: float = 0.1

    #Training
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0 

    #Model
    channels: int = 1

    #Logging / saving
    out_dir: str = "./runs_sr"
    save_every: int = 5

    # Visualization
    vis_every: int = 1              
    vis_max_items: int = 2         


def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ESPCN(nn.Module):
    def __init__(self, in_channels:int =1, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size = 5, padding = 2)
        self.conv2 = nn.Con2d(64,32,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, in_channels * (scale ** 2), kernel_size= 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = F.pixel_shuffle(x, self.scale)
        return x
    
#Metrics
@torch.no_grad()
def psnr( pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) **2, dim = (1,2,3))
    return 10.0 * torch.log10(1.0 / (mse + eps))

@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return torch.sqrt(mse + eps)


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target), dim=(1, 2, 3))


@torch.no_grad()
def mean_bias(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # mean(pred - target)
    return torch.mean(pred - target, dim=(1, 2, 3))


@torch.no_grad()
def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # R^2 per sample over all pixels/channels
    # 1 - SS_res / SS_tot
    y = target.reshape(target.shape[0], -1)
    yhat = pred.reshape(pred.shape[0], -1)
    ss_res = torch.sum((y - yhat) ** 2, dim=1)
    y_mean = torch.mean(y, dim=1, keepdim=True)
    ss_tot = torch.sum((y - y_mean) ** 2, dim=1)
    return 1.0 - (ss_res / (ss_tot + eps))

@torch.no_grad()
def pearson_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Pearson correlation per sample across all pixels/channels.
    Returns tensor of shape [B].
    """
    x = pred.reshape(pred.shape[0], -1)
    y = target.reshape(target.shape[0], -1)

    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    x0 = x - x_mean
    y0 = y - y_mean

    cov = (x0 * y0).mean(dim=1)
    x_std = torch.sqrt((x0 * x0).mean(dim=1) + eps)
    y_std = torch.sqrt((y0 * y0).mean(dim=1) + eps)

    return cov / (x_std * y_std + eps)


def _gaussian_kernel(window_size: int, sigma: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    # 1D Gaussian
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # Outer product -> 2D
    kernel_2d = torch.outer(g, g)
    return kernel_2d


@torch.no_grad()
def ssim_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    SSIM per sample, works for [B,C,H,W]. Uses a Gaussian window (depthwise conv).
    Assumes values are in [0, data_range].
    """
    assert pred.shape == target.shape
    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype

    # Make Gaussian window
    k = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    k = k.view(1, 1, window_size, window_size).repeat(C, 1, 1, 1)  # [C,1,ws,ws]

    # Depthwise conv
    padding = window_size // 2
    mu_x = F.conv2d(pred, k, padding=padding, groups=C)
    mu_y = F.conv2d(target, k, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, k, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, k, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, k, padding=padding, groups=C) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + eps)

    # Mean over C,H,W -> per sample
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def compute_metrics(sr: torch.Tensor, hr: torch.Tensor) -> dict:
    # clamp for metrics to keep them stable in [0,1]
    sr_c = sr.clamp(0, 1)
    hr_c = hr.clamp(0, 1)

    return {
        "psnr": psnr(sr_c, hr_c),
        "ssim": ssim_torch(sr_c, hr_c),
        "rmse": rmse(sr_c, hr_c),
        "mae": mae(sr_c, hr_c),
        "bias": mean_bias(sr_c, hr_c),
        "r2": r2_score(sr_c, hr_c),
        "pearson": pearson_corr(sr_c, hr_c)
    }


def _aggregate_metric_sums(agg: dict, batch_metrics: dict, bs: int) -> None:
    # batch_metrics values are tensors [B]
    for k, v in batch_metrics.items():
        agg[k] = agg.get(k, 0.0) + float(v.mean().item()) * bs


def _finalize_agg(agg: dict, n: int) -> dict:
    return {k: (v / max(n, 1)) for k, v in agg.items()}


# -----------------------------
# Visualization
# -----------------------------
@torch.no_grad()
def save_visualizations(
    lr_img: torch.Tensor,
    sr_img: torch.Tensor,
    hr_img: torch.Tensor,
    out_dir: str,
    epoch: int,
    scale: int,
    max_items: int = 2,
) -> None:
    """
    Saves LR, SR, HR, and error maps for a few items from a batch.
    Expected shapes:
      lr_img: [B,1,H/scale,W/scale]
      sr_img: [B,1,H,W]
      hr_img: [B,1,H,W]
    """
    os.makedirs(out_dir, exist_ok=True)

    B = lr_img.shape[0]
    count = min(B, max_items)

    # bicubic upsample LR to HR size for comparison

    lr_c = lr_img.clamp(0, 1)
    sr_c = sr_img.clamp(0, 1)
    hr_c = hr_img.clamp(0, 1)

    for i in range(count):
        lr_np = lr_c[i, 0].detach().cpu().numpy()
        sr_np = sr_c[i, 0].detach().cpu().numpy()
        hr_np = hr_c[i, 0].detach().cpu().numpy()
        err_np = np.abs(sr_np - hr_np)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(lr_np, cmap="inferno", vmin=0, vmax=1)
        axes[0].set_title("LR (input)")
        axes[1].imshow(sr_np, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("SR (model)")
        axes[2].imshow(hr_np, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("HR (target)")
        axes[3].imshow(err_np, cmap="magma")
        axes[3].set_title("|SR - HR|")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        path = os.path.join(out_dir, f"epoch_{epoch:03d}_sample_{i}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)

#train, eval loops
def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss, n = 0.0, 0
    agg = {}

    for lr_img, hr_img in loader:
        lr_img = lr_img.to(device)
        fr_img = hr_img.to(device)

        optimizer.zero_grad(set_to_none = True)
        sr = model(lr_img)

        loss = loss_fn(sr, hr_img)
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            metrics = compute_metrics(sr, hr_img)

        
        bs = lr_img.size(0)
        total_loss += loss.item() * bs
        _aggregate_metric_sums(agg, metrics, bs)
        n += bs

    if vis_batch is None:
            vis_batch = (lr_img.detach().cpu(), sr.detach().cpu(), hr_img.detach().cpu())

    return (total_loss / max(n, 1)), _finalize_agg(agg, n), vis_batch

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, n = 0.0, 0
    agg = {}

    for lr_img, hr_img in loader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        sr = model(lr_img)

        loss = loss_fn(sr, hr_img)
        metrics = compute_metrics(sr, hr_img)


        bs = lr_img.size(0)
        total_loss += loss.item() * bs
        _aggregate_metric_sums(agg, metrics, bs)
        n += bs

    if vis_batch is None:
            vis_batch = (lr_img.detach().cpu(), sr.detach().cpu(), hr_img.detach().cpu())

    return (total_loss / max(n, 1)), _finalize_agg(agg, n), vis_batch

def save_checkpoint(model, cfg: Config, epoch: int) -> str:
    os.makedirs(cfg.out_dir, exist_ok = True)
    path = os.path.join(cfg.out_dir, f"espcn_x{cfg.scale}_epoch{epoch}.pt")
    torch.save({"epoch": epoch, "model_state":model.state_dict(), "cfg": cfg.__dict__}, path)
    return path

def _fmt_metrics(d: dict) -> str:
    # compact formatting for printing
    return (
        f"psnr={d['psnr']:.2f} "
        f"ssim={d['ssim']:.4f} "
        f"rmse={d['rmse']:.4f} "
        f"mae={d['mae']:.4f} "
        f"bias={d['bias']:.4f} "
        f"r2={d['r2']:.4f}"
        f"pearson={d['pearson']:.4f}"
    )


def main():
    cfg = Config()
    set_seed(cfg.seed)

    #Dataset
    dataset = NightlightSRDataset( tif_folder = cfg.tif_folder,
                                  hr_size = cfg.hr_size,
                                  scale = cfg.scale,
                                  normalize = "log1p",
                                  clamp_max= 5.0
                                  )
    
    #train/val split
    val_len = max(1, int(len(dataset) * cfg.val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(
        train_ds,
        batch_size = cfg.batch_size,
        shuffle = True,
        num_worker = cfg.num_worker,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size= cfg.batch_size,
        shuffle=False,
        num_workers= cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


    model = ESPCN(in_channels=cfg.channels, scale=cfg.scale).to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay= cfg.weight_decay)
    loss_fn = nn.L1Loss()


    print(f"Device: {cfg.device}")
    print(f"Samples: train={train_len}, val={val_len}")
    print(f"SR scale: x{cfg.scale}, HR size: {cfg.hr_size}, Channels: {cfg.channels}")
    print(f"Outputs: {os.path.abspath(cfg.out_dir)}")

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, cfg.device, loss_fn)
        val_loss, val_m, vis_batch = evaluate(model, val_loader, cfg.device, loss_fn)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.5f} { _fmt_metrics(train_m) } | "
            f"val_loss={val_loss:.5f} { _fmt_metrics(val_m) }"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = save_checkpoint(model, cfg, epoch)
            print(f"  ✓ saved best checkpoint: {ckpt}")

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            ckpt = save_checkpoint(model, cfg, epoch)
            print(f"  saved periodic checkpoint: {ckpt}")

        # Visualizations
        if cfg.vis_every > 0 and epoch % cfg.vis_every == 0 and vis_batch is not None:
            vis_dir = os.path.join(cfg.out_dir, "visuals")
            lr_b, sr_b, hr_b = vis_batch
            save_visualizations(
                lr_img=lr_b,
                sr_img=sr_b,
                hr_img=hr_b,
                out_dir=vis_dir,
                epoch=epoch,
                scale=cfg.scale,
                max_items=cfg.vis_max_items,
            )
            print(f"  saved visualizations to: {os.path.abspath(vis_dir)}")



if __name__ == "__main__":
    main()