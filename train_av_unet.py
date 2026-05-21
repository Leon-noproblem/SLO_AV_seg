import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import AVDataset
from model_unet import UNet


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def estimate_class_weights(ds):
    cnt = np.zeros(3, dtype=np.float64)
    for i in range(min(len(ds), 200)):
        y = ds[i]["label"].numpy()
        for c in [1, 2]:
            cnt[c] += np.sum(y == c)
    cnt = np.maximum(cnt, 1.0)
    w = np.array([0.0, 1.0 / cnt[1], 1.0 / cnt[2]], dtype=np.float32)
    w = w / (w[1:].mean() + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)
            y = b["label"].to(device)
            pred = torch.argmax(net(x), dim=1)
            m = y > 0
            correct += (pred[m] == y[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = AVDataset(args.data_root, "train", with_label=True, use_rgb=(args.in_ch==4))
    n_val = max(1, int(len(full_train) * args.val_ratio))
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    net = UNet(in_ch=args.in_ch, out_ch=3, base=args.base).to(device)
    class_w = estimate_class_weights(full_train).to(device)
    ce = nn.CrossEntropyLoss(ignore_index=0, weight=class_w)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for ep in range(args.epochs):
        net.train()
        losses = []
        for b in train_loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = net(x)
                loss = ce(logits, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        sch.step()
        acc = evaluate(net, val_loader, device)
        print(f"epoch={ep+1:03d} loss={np.mean(losses):.5f} val_acc={acc:.5f} lr={opt.param_groups[0]['lr']:.2e}")
        if acc > best:
            best = acc
            torch.save({"model": net.state_dict(), "epoch": ep + 1, "acc": acc, "args": vars(args)}, ckpt_dir / "best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--in_ch", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_dir", default="checkpoints")
    main(p.parse_args())
