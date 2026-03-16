import rasterio
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

#input: TIF files 
#output: Dataset of downsampled and pre-processed NTL images
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
            random_crop: bool = True,

    ):
        # get TIF files folder
        self.files = sorted(Path(tif_folder).glob("*.tif"))
        if not self.files:
            raise FileNotFoundError(f"No .tif files found in {tif_folder}")
        if hr_size % scale != 0:
            raise ValueError("hr_size must be divisible by scale")
        
        self.hr_size = hr_size
        self.scale = scale
        self.normalize = normalize
        self.clamp_max = clamp_max
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)
    
    def _read_tif(self, path: Path) -> np.ndarray:
        # read TIF files
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodata
        # set non accepted values to 0.0
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
    
    #random crop the image for the training data 
    def _random_crop_tensor(self, t:torch.Tensor) -> torch.Tensor:
        t = self._check_min_size_tensor(t, self.hr_size, self.hr_size)
        _, h,w = t.shape

        top = torch.randint(0, h - self.hr_size + 1, (1,)).item()
        left = torch.randint(0, w - self.hr_size + 1, (1,)).item()

        return t[:, top:top + self.hr_size, left:left + self.hr_size]
    
    #deterministic crop for the validation data
    def _center_crop_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t = self._check_min_size_tensor(t, self.hr_size, self.hr_size)
        _, h, w = t.shape

        top = max((h - self.hr_size) // 2, 0)
        left = max((w - self.hr_size) // 2, 0)

        return t[:, top:top + self.hr_size, left:left + self.hr_size]
    
    def __getitem__(self, idx):
        path = self.files[idx]
        image_id = path.stem
        hr_np = self._read_tif(path)
        hr_np = self._normalize(hr_np)

        hr_t = torch.from_numpy(hr_np).float().unsqueeze(0)
        
        if self.random_crop:
            hr_t = self._random_crop_tensor(hr_t)
        else:
            hr_t = self._center_crop_tensor(hr_t)
        sigma = float(0.5 * self.scale) #1 if scale = 2
       
       #downsampling
        lr_t = TF.gaussian_blur(hr_t.unsqueeze(0), kernel_size=5, sigma=sigma)
        lr_t = F.interpolate(
            lr_t,
            scale_factor = 1 / self.scale,
            mode = "bicubic",
            align_corners = False
        ).squeeze(0)

        return lr_t, hr_t, image_id




