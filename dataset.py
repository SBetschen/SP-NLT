import rasterio
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset

class NightlightSRDataset(Dataset):
    """
    Returns:
        lr: [1, hr_size/scale, hr_size/scale]
        hr: [1, hr_size, hr_size]
    """
    def __init__(
            self,
            tif_folder: str,
            hr_size: int = 128,
            scale: int = 2,
            normalize: str = "log1p",
            clamp_max: float = 5.0,

    ):
        self.files = sorted(Path(tif_folder).glob("*.tif"))
        if not self.files:
            raise FileNotFoundError(f"No .tif files found in {tif_folder}")
        if hr_size % scale != 0:
            raise ValueError("hr_size must be divisible by scale")
        
        self.hr_size = hr_size
        self.scale = scale
        self.normalize = normalize
        self.clamp_max = clamp_max

    def __len__(self):
        return len(self.files)
    
    def _read_tif(self, path: Path) -> np.ndarray:

        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodata

        if nodata is not None:
            img = np.where(img == nodata, 0.0, img)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.clip(img, 0.0, None)

        return img

    def _normalize(self, x :np.ndarray) -> np.ndarray:
        x = np.log1p(x)
        m = x.max()
        return x/m if m>0 else x
    
    @staticmethod
    def _check_min_size_tensor(t: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
        _, h, w = t.shape
        pad_h = max(min_h - h,0)
        pad_w = max(min_w -w,0)
        if pad_h == 0 and pad_w == 0:
            return t
        return F.pad(t, (0,pad_w,0,pad_h), mode="constant", value=0.0)
    
    def _random_crop_tensor(self, t:torch.Tensor) -> torch.Tensor:
        t = self._check_min_size_tensor(t, self.hr_size, self.hr_size)
        _, h,w = t.shape

        top = torch.randint(0, h - self.hr_size + 1, (1,)).item()
        left = torch.randint(0, w - self.hr_size + 1, (1,)).item()

        return t[:, top:top + self.hr_size, left:left + self.hr_size]
    
    def __getitem__(self, idx):
        hr_np = self._read_tif(self.files[idx])
        hr_np = self._normalize(hr_np)

        hr_t = torch.from_numpy(hr_np).float().unsqueeze(0)
        hr_t = self._random_crop_tensor(hr_t)

        lr_t = F.interpolate(
            hr_t.unsqueeze(0),
            scale_factor = 1 / self.scale,
            mode = "bicubic",
            align_corners = False
        ).squeeze(0)

        return lr_t, hr_t



def center_crop(arr, out_h=128, out_w=128):
    h,w = arr.shape
    top = max((h - out_h) // 2,0)
    left = max((w - out_w) //2, 0)
    return arr[top:top+out_h, left:left+out_w]

ds = NightlightSRDataset(tif_folder=".")
lr, hr = ds[0]

print("HR shape:", hr.shape)
print("LR shape:", lr.shape)

import matplotlib.pyplot as plt

plt.imshow(lr[0], cmap="inferno", vmin=0, vmax=1)
plt.colorbar(label="Nightlight intensity")
plt.title("Nighttime Lights")
plt.show()