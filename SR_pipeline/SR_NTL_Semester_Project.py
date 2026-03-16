# -*- coding: utf-8 -*-
import os
import csv
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from dataset import NightlightSRDataset


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Data
    tif_folder: str = "./data"
    hr_size: int = 128
    scale: int = 2
    batch_size: int = 16
    num_workers: int = 2
    val_split: float = 0.1

    # Training
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Model
    channels: int = 1

    # Logging / saving
    out_dir: str = "./runs_sr"

    # Visualization
    vis_every: int = 1
    vis_num_items: int = 4  # fixed random validation images


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _aggregate_metric_sums(agg: dict, batch_metrics: dict, bs: int) -> None:
    for k, v in batch_metrics.items():
        agg[k] = agg.get(k, 0.0) + float(v.mean().item()) * bs


def _finalize_agg(agg: dict, n: int) -> dict:
    return {k: (v / max(n, 1)) for k, v in agg.items()}


# -----------------------------
# Logging / saving
# -----------------------------
def save_checkpoint(model, cfg: Config, epoch: int) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, "best_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "cfg": cfg.__dict__,
        },
        path,
    )
    return path


def append_val_metrics_csv(cfg: Config, epoch: int, train_loss: float, val_loss: float, val_m: dict) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    csv_path = os.path.join(cfg.out_dir, "val_metrics.csv")
    file_exists = os.path.isfile(csv_path)

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_psnr": val_m.get("psnr"),
        "val_ssim": val_m.get("ssim"),
        "val_rmse": val_m.get("rmse"),
        "val_mae": val_m.get("mae"),
        "val_bias": val_m.get("bias"),
        "val_r2": val_m.get("r2"),
        "val_pearson": val_m.get("pearson"),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _fmt_metrics(d: dict) -> str:
    def g(k, fmt):
        v = d.get(k, None)
        return "nan" if v is None else format(v, fmt)

    return (
        f"psnr={g('psnr', '.2f')} "
        f"ssim={g('ssim', '.4f')} "
        f"rmse={g('rmse', '.4f')} "
        f"mae={g('mae', '.4f')} "
        f"bias={g('bias', '.4f')} "
        f"r2={g('r2', '.4f')} "
        f"pearson={g('pearson', '.4f')}"
    )


# -----------------------------
# Visualization
# -----------------------------
@torch.no_grad()
def save_fixed_val_panel(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    out_dir: str,
    epoch: int,
) -> None:
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for lr_img, hr_img, image_ids in loader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        sr_img = model(lr_img).clamp(0, 1)

        rows.append(
            (
                lr_img.detach().cpu(),
                sr_img.detach().cpu(),
                hr_img.detach().cpu(),
                list(image_ids),
            )
        )

    if not rows:
        return

    total_items = sum(batch[0].shape[0] for batch in rows)
    fig, axes = plt.subplots(total_items, 4, figsize=(16, 4 * total_items))

    if total_items == 1:
        axes = np.expand_dims(axes, axis=0)

    row_idx = 0
    for lr_b, sr_b, hr_b, ids_b in rows:
        batch_size = lr_b.shape[0]
        for i in range(batch_size):
            lr_np = lr_b[i, 0].numpy()
            sr_np = sr_b[i, 0].numpy()
            hr_np = hr_b[i, 0].numpy()
            err_np = np.abs(sr_np - hr_np)

            axes[row_idx, 0].imshow(lr_np, cmap="inferno", vmin=0, vmax=1)
            axes[row_idx, 0].set_title(f"{ids_b[i]}\nLR")

            axes[row_idx, 1].imshow(sr_np, cmap="inferno", vmin=0, vmax=1)
            axes[row_idx, 1].set_title("SR")

            axes[row_idx, 2].imshow(hr_np, cmap="inferno", vmin=0, vmax=1)
            axes[row_idx, 2].set_title("HR")

            axes[row_idx, 3].imshow(err_np, cmap="magma")
            axes[row_idx, 3].set_title("|SR - HR|")

            for ax in axes[row_idx]:
                ax.axis("off")

            row_idx += 1

    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Train / eval
# -----------------------------
def train_one_epoch(cfg: Config, model, loader, optimizer, device, loss_fn):
    model.train()

    total_loss, n = 0.0, 0
    agg = {}

    for lr_img, hr_img, _ in loader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        optimizer.zero_grad(set_to_none=True)
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

    return (total_loss / max(n, 1)), _finalize_agg(agg, n)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()

    total_loss, n = 0.0, 0
    agg = {}

    for lr_img, hr_img, _ in loader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        sr = model(lr_img)
        loss = loss_fn(sr, hr_img)
        metrics = compute_metrics(sr, hr_img)

        bs = lr_img.size(0)
        total_loss += loss.item() * bs
        _aggregate_metric_sums(agg, metrics, bs)
        n += bs

    return (total_loss / max(n, 1)), _finalize_agg(agg, n)

def save_learning_curves(cfg: Config) -> None:
    csv_path = os.path.join(cfg.out_dir, "val_metrics.csv")
    if not os.path.isfile(csv_path):
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    if data.size == 0:
        return

    # Handle single-row CSV
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    epochs = data["epoch"]

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, data["train_loss"], label="train_loss")
    plt.plot(epochs, data["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "learning_curves_loss.png"), dpi=150)
    plt.close()

    # Metric curve example: PSNR
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, data["val_psnr"], label="val_psnr")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("Validation PSNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "learning_curves_psnr.png"), dpi=150)
    plt.close()

def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    val_csv_path = os.path.join(cfg.out_dir, "val_metrics.csv")
    if os.path.exists(val_csv_path):
        os.remove(val_csv_path)

    # Datasets
    full_train_dataset = NightlightSRDataset(
        tif_folder=cfg.tif_folder,
        hr_size=cfg.hr_size,
        scale=cfg.scale,
        normalize="log1p",
        clamp_max=5.0,
        random_crop=True,
    )

    full_val_dataset = NightlightSRDataset(
        tif_folder=cfg.tif_folder,
        hr_size=cfg.hr_size,
        scale=cfg.scale,
        normalize="log1p",
        clamp_max=5.0,
        random_crop=False,
    )

    n_total = len(full_train_dataset)
    n_val = max(1, int(n_total * cfg.val_split))
    n_train = n_total - n_val

    if n_train <= 0:
        raise ValueError("Not enough samples for train/val split.")

    rng = np.random.default_rng(cfg.seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = Subset(full_train_dataset, train_indices)
    val_ds = Subset(full_val_dataset, val_indices)

    # Fixed random validation images for visualization
    n_vis = min(cfg.vis_num_items, len(val_indices))
    vis_indices = rng.choice(val_indices, size=n_vis, replace=False).tolist()
    vis_ds = Subset(full_val_dataset, vis_indices)

    torch.save(
        {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "vis_indices": vis_indices,
        },
        os.path.join(cfg.out_dir, "split_indices.pt"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    vis_loader = DataLoader(
        vis_ds,
        batch_size=n_vis,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = ESPCN(in_channels=cfg.channels, scale=cfg.scale).to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.L1Loss()

    print(f"Device: {cfg.device}")
    print(f"Samples: train={n_train}, val={n_val}")
    print(f"Fixed validation visuals: {n_vis}")
    print(f"SR scale: x{cfg.scale}, HR size: {cfg.hr_size}, Channels: {cfg.channels}")
    print(f"Outputs: {os.path.abspath(cfg.out_dir)}")

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_m = train_one_epoch(cfg, model, train_loader, optimizer, cfg.device, loss_fn)
        val_loss, val_m = evaluate(model, val_loader, cfg.device, loss_fn)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.5f} {_fmt_metrics(train_m)} | "
            f"val_loss={val_loss:.5f} {_fmt_metrics(val_m)}"
        )

        append_val_metrics_csv(cfg, epoch, train_loss, val_loss, val_m)
        save_learning_curves(cfg)

        if val_loss < best_val:
            best_val = val_loss
            ckpt = save_checkpoint(model, cfg, epoch)
            print(f"  saved best checkpoint: {ckpt}")

        if cfg.vis_every > 0 and epoch % cfg.vis_every == 0:
            vis_dir = os.path.join(cfg.out_dir, "val_visuals")
            save_fixed_val_panel(
                model=model,
                loader=vis_loader,
                device=cfg.device,
                out_dir=vis_dir,
                epoch=epoch,
            )
            print(f"  saved fixed validation panel to: {os.path.abspath(vis_dir)}")


if __name__ == "__main__":
    main()