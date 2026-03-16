# test.py
# Runs test evaluation for:
# 1) learned SR model from best_model.pt
# 2) bicubic interpolation baseline
# Saves:
# - one CSV per method
# - one visualization per test image per method

# -*- coding: utf-8 -*-
import os
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NightlightSRDataset


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    test_folder: str = "./test_data"  
    hr_size: int = 128
    scale: int = 2
    batch_size: int = 16
    num_workers: int = 2

    # Model / paths
    channels: int = 1
    out_dir: str = "./runs_sr"
    checkpoint_name: str = "best_model.pt"


class ESPCN(nn.Module):
    def __init__(self, in_channels: int = 1, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, in_channels * (scale ** 2), kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = F.pixel_shuffle(x, self.scale)
        return x


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
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
    return torch.mean(pred - target, dim=(1, 2, 3))


@torch.no_grad()
def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    y = target.reshape(target.shape[0], -1)
    yhat = pred.reshape(pred.shape[0], -1)
    ss_res = torch.sum((y - yhat) ** 2, dim=1)
    y_mean = torch.mean(y, dim=1, keepdim=True)
    ss_tot = torch.sum((y - y_mean) ** 2, dim=1)
    return 1.0 - (ss_res / (ss_tot + eps))


@torch.no_grad()
def pearson_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


@torch.no_grad()
def ssim_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    assert pred.shape == target.shape
    _, c, _, _ = pred.shape
    device = pred.device
    dtype = pred.dtype

    k = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    k = k.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)

    padding = window_size // 2
    mu_x = F.conv2d(pred, k, padding=padding, groups=c)
    mu_y = F.conv2d(target, k, padding=padding, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, k, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, k, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, k, padding=padding, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + eps)

    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def compute_metrics(sr: torch.Tensor, hr: torch.Tensor) -> dict:
    sr_c = sr.clamp(0, 1)
    hr_c = hr.clamp(0, 1)

    return {
        "psnr": psnr(sr_c, hr_c),
        "ssim": ssim_torch(sr_c, hr_c),
        "rmse": rmse(sr_c, hr_c),
        "mae": mae(sr_c, hr_c),
        "bias": mean_bias(sr_c, hr_c),
        "r2": r2_score(sr_c, hr_c),
        "pearson": pearson_corr(sr_c, hr_c),
    }


# -----------------------------
# SR methods
# -----------------------------
@torch.no_grad()
def bicubic_sr(lr_img: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(
        lr_img,
        scale_factor=scale,
        mode="bicubic",
        align_corners=False
    )


# -----------------------------
# CSV helpers
# -----------------------------
def init_metrics_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "psnr",
                "ssim",
                "rmse",
                "mae",
                "bias",
                "r2",
                "pearson",
            ],
        )
        writer.writeheader()


def append_metrics_rows(csv_path: str, image_ids, batch_metrics: dict) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "psnr",
                "ssim",
                "rmse",
                "mae",
                "bias",
                "r2",
                "pearson",
            ],
        )

        for i in range(len(image_ids)):
            writer.writerow(
                {
                    "image_id": image_ids[i],
                    "psnr": float(batch_metrics["psnr"][i].item()),
                    "ssim": float(batch_metrics["ssim"][i].item()),
                    "rmse": float(batch_metrics["rmse"][i].item()),
                    "mae": float(batch_metrics["mae"][i].item()),
                    "bias": float(batch_metrics["bias"][i].item()),
                    "r2": float(batch_metrics["r2"][i].item()),
                    "pearson": float(batch_metrics["pearson"][i].item()),
                }
            )


# -----------------------------
# Visualization
# -----------------------------
@torch.no_grad()
def save_batch_visualizations(
    lr_img: torch.Tensor,
    sr_img: torch.Tensor,
    hr_img: torch.Tensor,
    image_ids,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    lr_c = lr_img.clamp(0, 1).detach().cpu()
    sr_c = sr_img.clamp(0, 1).detach().cpu()
    hr_c = hr_img.clamp(0, 1).detach().cpu()

    for i in range(lr_c.shape[0]):
        lr_np = lr_c[i, 0].numpy()
        sr_np = sr_c[i, 0].numpy()
        hr_np = hr_c[i, 0].numpy()
        err_np = np.abs(sr_np - hr_np)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(lr_np, cmap="inferno", vmin=0, vmax=1)
        axes[0].set_title("LR")

        axes[1].imshow(sr_np, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("SR")

        axes[2].imshow(hr_np, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("HR")

        axes[3].imshow(err_np, cmap="magma")
        axes[3].set_title("|SR - HR|")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{image_ids[i]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# -----------------------------
# Evaluation runners
# -----------------------------
@torch.no_grad()
def run_learned_test(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    csv_path: str,
    vis_dir: str,
) -> None:
    model.eval()
    init_metrics_csv(csv_path)

    for lr_img, hr_img, image_ids in loader:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        sr_img = model(lr_img)
        metrics = compute_metrics(sr_img, hr_img)

        append_metrics_rows(csv_path, image_ids, metrics)
        save_batch_visualizations(lr_img, sr_img, hr_img, image_ids, vis_dir)


@torch.no_grad()
def run_bicubic_test(
    loader: DataLoader,
    device: str,
    scale: int,
    csv_path: str,
    vis_dir: str,
) -> None:
    init_metrics_csv(csv_path)

    for lr_img, hr_img, image_ids in loader:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        sr_img = bicubic_sr(lr_img, scale=scale)
        metrics = compute_metrics(sr_img, hr_img)

        append_metrics_rows(csv_path, image_ids, metrics)
        save_batch_visualizations(lr_img, sr_img, hr_img, image_ids, vis_dir)


def main():
    cfg = Config()

    ckpt_path = os.path.join(cfg.out_dir, cfg.checkpoint_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=cfg.device)

    ckpt_cfg = checkpoint.get("cfg", {})
    scale = int(ckpt_cfg.get("scale", cfg.scale))
    channels = int(ckpt_cfg.get("channels", cfg.channels))
    hr_size = int(ckpt_cfg.get("hr_size", cfg.hr_size))

    model = ESPCN(in_channels=channels, scale=scale).to(cfg.device)
    model.load_state_dict(checkpoint["model_state"])

    test_dataset = NightlightSRDataset(
        tif_folder=cfg.test_folder,
        hr_size=hr_size,
        scale=scale,
        normalize="log1p",
        clamp_max=5.0,
        random_crop=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    learned_csv = os.path.join(cfg.out_dir, "test_metrics_learned.csv")
    bicubic_csv = os.path.join(cfg.out_dir, "test_metrics_bicubic.csv")

    learned_vis_dir = os.path.join(cfg.out_dir, "test_visuals_learned")
    bicubic_vis_dir = os.path.join(cfg.out_dir, "test_visuals_bicubic")

    run_learned_test(
        model=model,
        loader=test_loader,
        device=cfg.device,
        csv_path=learned_csv,
        vis_dir=learned_vis_dir,
    )
    print(f"Saved learned-model metrics to: {os.path.abspath(learned_csv)}")
    print(f"Saved learned-model visualizations to: {os.path.abspath(learned_vis_dir)}")

    run_bicubic_test(
        loader=test_loader,
        device=cfg.device,
        scale=scale,
        csv_path=bicubic_csv,
        vis_dir=bicubic_vis_dir,
    )
    print(f"Saved bicubic metrics to: {os.path.abspath(bicubic_csv)}")
    print(f"Saved bicubic visualizations to: {os.path.abspath(bicubic_vis_dir)}")


if __name__ == "__main__":
    main()