from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _map_av_rgb_to_train_ids(av_bgr: np.ndarray) -> np.ndarray:
    """Map AV RGB label image to train ids {0,1,2}.

    Rules (after /255):
    1) vessel if any channel > 0.5
    2) otherwise background
    3) if all channels > 0.5 (white), treat as background
    4) R > 0.5 => artery (1)
    5) B > 0.5 => vein (2)

    Priority for ambiguous pixels (both R and B > 0.5, but not white):
    - choose stronger channel; tie -> background.
    """
    av = av_bgr.astype(np.float32) / 255.0
    b = av[..., 0]
    g = av[..., 1]
    r = av[..., 2]

    any_on = (r > 0.5) | (g > 0.5) | (b > 0.5)
    all_on = (r > 0.5) & (g > 0.5) & (b > 0.5)

    y = np.zeros(av.shape[:2], dtype=np.int64)

    vessel = any_on & (~all_on)
    artery = vessel & (r > 0.5)
    vein = vessel & (b > 0.5)

    # resolve ambiguity where artery and vein both true
    both = artery & vein
    artery_only = artery & (~both)
    vein_only = vein & (~both)

    y[artery_only] = 1
    y[vein_only] = 2

    # both red and blue active but not white: choose stronger channel, tie -> bg
    y[both & (r > b)] = 1
    y[both & (b > r)] = 2

    return y


class AVDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", with_label: bool = True, use_rgb: bool = True):
        self.root = Path(data_root) / split
        self.img_dir = self.root / "image"
        self.mask_dir = self.root / "mask"
        self.av_dir = self.root / "av"
        self.with_label = with_label
        self.use_rgb = use_rgb
        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _index_by_stem(folder: Path):
        d = {}
        for p in folder.iterdir():
            if p.is_file():
                d[p.stem.lower()] = p
        return d

    def _build_samples(self):
        img_idx = self._index_by_stem(self.img_dir)
        mask_idx = self._index_by_stem(self.mask_dir)
        av_idx = self._index_by_stem(self.av_dir) if self.with_label else {}

        common = sorted(set(img_idx.keys()) & set(mask_idx.keys()))
        if self.with_label:
            common = sorted(set(common) & set(av_idx.keys()))

        if not common:
            raise RuntimeError(f"No matched samples in {self.root}. Check filenames/stems across image/mask/av.")

        samples = []
        for k in common:
            item = {"name": img_idx[k].name, "img": img_idx[k], "mask": mask_idx[k]}
            if self.with_label:
                item["av"] = av_idx[k]
            samples.append(item)

        miss_mask = sorted(set(img_idx.keys()) - set(mask_idx.keys()))
        miss_img = sorted(set(mask_idx.keys()) - set(img_idx.keys()))
        miss_av = sorted(set(common) - set(av_idx.keys())) if self.with_label else []
        if miss_mask or miss_img or miss_av:
            print(f"[AVDataset] matched={len(samples)} miss_mask={len(miss_mask)} miss_img={len(miss_img)} miss_av={len(miss_av)}")
        return samples

    @staticmethod
    def _read_gray(path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    @staticmethod
    def _read_color(path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _read_img(self, path: Path):
        if self.use_rgb:
            img = self._read_color(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
        else:
            img = self._read_gray(path).astype(np.float32) / 255.0
            img = img[None, ...]
        return img

    def __getitem__(self, idx):
        item = self.samples[idx]
        name = item["name"]
        img = self._read_img(item["img"])
        mask = self._read_gray(item["mask"])
        mask = (mask > 127).astype(np.float32)
        x = np.concatenate([img, mask[None, ...]], axis=0)

        sample = {
            "name": name,
            "image": torch.from_numpy(x).float(),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
        }

        if self.with_label:
            av = self._read_color(item["av"])
            y = _map_av_rgb_to_train_ids(av)
            y[mask == 0] = 0
            sample["label"] = torch.from_numpy(y)
        return sample
