# SLO AV Segmentation (Mask-Guided + Topology Propagation)

本实现针对 1000x1000 PNG 眼底图像，输入为 `image + mask`，不再使用 IterNet，直接进行动静脉分类并使用拓扑传播后处理。

## 数据结构

```
SLO_AV_dataset/
  train/
    image/
    mask/
    av/
  test/
    image/
    mask/
    av/
```

## 方法概述

1. **分类网络（Full-size U-Net）**
   - 输入通道默认 4（RGB + mask）
   - 输出类别：0 背景，1 动脉，2 静脉

2. **拓扑传播后处理（更接近 SeqNet 论文）**
   - 二值化血管后提取 skeleton
   - 检测关键点（终点 + 交叉/分叉点）
   - 两关键点之间定义 segment
   - 每个 segment 计算整体 artery/vein 置信度并统一标签
   - inter-segment prediction propagation：
     - 方向关系 \(A_{ij}\)
     - 连接线关系 \(L_{ij}\)
     - 厚度相似性 \(T_{ij}\)
     - 距离项 \(D_{ij}\)
     - 综合影响系数 \(W_{ij}=A_{ij}L_{ij}T_{ij}D_{ij}\)

## 训练建议（已优化参数）

- `AdamW + CosineAnnealingLR`
- 类别权重平衡（artery/vein）
- AMP 混合精度
- 默认参数适配 1000x1000：

```bash
python train_av_unet.py \
  --data_root SLO_AV_dataset \
  --in_ch 4 --base 32 \
  --batch_size 1 --epochs 120 \
  --lr 3e-4 --wd 1e-4 \
  --val_ratio 0.2 --num_workers 4
```

## 推理

```bash
python predict_av_unet.py \
  --data_root SLO_AV_dataset \
  --split test \
  --ckpt checkpoints/best.pt \
  --in_ch 4 --base 32 \
  --out_dir output
```


## 一键训练（服务器直接运行）

```bash
bash one_click_train.sh /path/to/SLO_AV_dataset
```

可选环境变量（不传则用默认值）：

```bash
EPOCHS=120 BATCH_SIZE=1 LR=3e-4 WD=1e-4 BASE=32 IN_CH=4 NUM_WORKERS=4 VAL_RATIO=0.2 \
bash one_click_train.sh /path/to/SLO_AV_dataset
```
