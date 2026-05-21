from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class AVDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", with_label: bool = True, use_rgb: bool = True):
        self.root = Path(data_root) / split
        self.img_dir = self.root / "image"
        self.mask_dir = self.root / "mask"
        self.av_dir = self.root / "av"
        self.with_label = with_label
        self.use_rgb = use_rgb
        self.names = sorted([p.name for p in self.img_dir.iterdir() if p.is_file()])

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _read_gray(path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _read_img(self, path: Path):
        if self.use_rgb:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
        else:
            img = self._read_gray(path).astype(np.float32) / 255.0
            img = img[None, ...]
        return img

    def __getitem__(self, idx):
        name = self.names[idx]
        img = self._read_img(self.img_dir / name)
        mask = self._read_gray(self.mask_dir / name)
        mask = (mask > 127).astype(np.float32)
        x = np.concatenate([img, mask[None, ...]], axis=0)

        sample = {
            "name": name,
            "image": torch.from_numpy(x).float(),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
        }

        if self.with_label:
            av = self._read_gray(self.av_dir / name)
            y = av.astype(np.int64)
            y[mask == 0] = 0
            sample["label"] = torch.from_numpy(y)
        return sample
