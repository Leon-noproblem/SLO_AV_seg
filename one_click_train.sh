#!/usr/bin/env bash
set -euo pipefail

# =========================
# One-click AV pipeline
# =========================
# Usage:
#   bash one_click_train.sh /path/to/SLO_AV_dataset
# Optional env:
#   EPOCHS=120 BATCH_SIZE=1 LR=3e-4 WD=1e-4 BASE=32 IN_CH=4 NUM_WORKERS=4 VAL_RATIO=0.2

DATA_ROOT="${1:-SLO_AV_dataset}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-3e-4}"
WD="${WD:-1e-4}"
BASE="${BASE:-32}"
IN_CH="${IN_CH:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUT_DIR="${OUT_DIR:-output}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/6] Python version"
$PYTHON_BIN --version

echo "[2/6] Install dependencies"
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install --no-cache-dir numpy opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
$PYTHON_BIN -m pip install --no-cache-dir numpy opencv-python torch torchvision torchaudio

echo "[3/6] Check dataset path: $DATA_ROOT"
for d in \
  "$DATA_ROOT/train/image" "$DATA_ROOT/train/mask" "$DATA_ROOT/train/av" \
  "$DATA_ROOT/test/image"  "$DATA_ROOT/test/mask"; do
  if [[ ! -d "$d" ]]; then
    echo "[ERROR] Missing directory: $d"
    exit 1
  fi
done

TRAIN_COUNT=$(find "$DATA_ROOT/train/image" -maxdepth 1 -type f | wc -l)
TEST_COUNT=$(find "$DATA_ROOT/test/image" -maxdepth 1 -type f | wc -l)
if [[ "$TRAIN_COUNT" -eq 0 || "$TEST_COUNT" -eq 0 ]]; then
  echo "[ERROR] Empty dataset folders. train=$TRAIN_COUNT test=$TEST_COUNT"
  exit 1
fi
echo "[OK] train images: $TRAIN_COUNT, test images: $TEST_COUNT"

echo "[4/6] Train"
$PYTHON_BIN train_av_unet.py \
  --data_root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --wd "$WD" \
  --base "$BASE" \
  --in_ch "$IN_CH" \
  --val_ratio "$VAL_RATIO" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --ckpt_dir "$CKPT_DIR"

if [[ ! -f "$CKPT_DIR/best.pt" ]]; then
  echo "[ERROR] Training finished but checkpoint not found: $CKPT_DIR/best.pt"
  exit 1
fi

echo "[5/6] Inference + internal postprocess"
$PYTHON_BIN predict_av_unet.py \
  --data_root "$DATA_ROOT" \
  --split test \
  --ckpt "$CKPT_DIR/best.pt" \
  --out_dir "$OUT_DIR" \
  --base "$BASE" \
  --in_ch "$IN_CH"

echo "[6/6] Standalone postprocess (re-run to ensure reproducible final folder)"
$PYTHON_BIN postprocess_av.py \
  --pred_dir "$OUT_DIR/raw" \
  --mask_dir "$DATA_ROOT/test/mask" \
  --out_dir "$OUT_DIR/post"

echo "Done. Outputs:"
echo "  Raw:  $OUT_DIR/raw"
echo "  Post: $OUT_DIR/post"
echo "  Ckpt: $CKPT_DIR/best.pt"
