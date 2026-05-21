import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import AVDataset
from model_unet import UNet
from postprocess_av import postprocess


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AVDataset(args.data_root, args.split, with_label=False, use_rgb=(args.in_ch==4))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    net = UNet(in_ch=args.in_ch, out_ch=3, base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    out_raw = Path(args.out_dir) / "raw"
    out_post = Path(args.out_dir) / "post"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_post.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)
            m = b["mask"].squeeze().cpu().numpy().astype(np.uint8)
            name = b["name"][0]

            logits = net(x)
            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            pred[m == 0] = 0
            cv2.imwrite(str(out_raw / name), pred)

            post = postprocess(pred, m)
            cv2.imwrite(str(out_post / name), post.astype(np.uint8))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="output")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--in_ch", type=int, default=4)
    main(p.parse_args())
