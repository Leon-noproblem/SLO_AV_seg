import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Segment:
    sid: int
    pixels: np.ndarray  # (N, 2) -> y,x
    p0: Tuple[int, int]
    p1: Tuple[int, int]
    t0: np.ndarray
    t1: np.ndarray
    width: float


def _neighbors8(y, x, h, w):
    for yy in range(max(0, y - 1), min(h, y + 2)):
        for xx in range(max(0, x - 1), min(w, x + 2)):
            if yy == y and xx == x:
                continue
            yield yy, xx


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    skel = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    img = mask.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = cv2.countNonZero(img) == 0
    return skel


def _keypoints(skel: np.ndarray):
    h, w = skel.shape
    deg = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        c = 0
        for yy, xx in _neighbors8(y, x, h, w):
            if skel[yy, xx] > 0:
                c += 1
        deg[y, x] = c
    endpoints = (skel > 0) & (deg == 1)
    junctions = (skel > 0) & (deg >= 3)
    key = endpoints | junctions
    return endpoints, junctions, key


def _estimate_tangent(pixels: np.ndarray, at_start: bool, n=7):
    if len(pixels) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    if at_start:
        q0 = pixels[0]
        q1 = pixels[min(len(pixels) - 1, n - 1)]
        v = (q1 - q0).astype(np.float32)
    else:
        q0 = pixels[-1]
        q1 = pixels[max(0, len(pixels) - n)]
        v = (q0 - q1).astype(np.float32)
    norm = np.linalg.norm(v) + 1e-6
    return v / norm


def extract_segments(vessel_mask: np.ndarray):
    skel = _skeletonize(vessel_mask)
    _, _, key = _keypoints(skel)
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=np.uint8)
    segs: List[Segment] = []
    sid = 0

    key_pts = list(zip(*np.where(key > 0)))
    for ky, kx in key_pts:
        for ny, nx in _neighbors8(ky, kx, h, w):
            if skel[ny, nx] == 0 or visited[ny, nx] > 0:
                continue
            path = [(ky, kx)]
            py, px = ky, kx
            cy, cx = ny, nx
            visited[cy, cx] = 1
            path.append((cy, cx))
            while True:
                if key[cy, cx] and (cy, cx) != (ky, kx):
                    break
                cands = []
                for ty, tx in _neighbors8(cy, cx, h, w):
                    if skel[ty, tx] == 0:
                        continue
                    if (ty, tx) == (py, px):
                        continue
                    cands.append((ty, tx))
                if len(cands) == 0:
                    break
                if len(cands) > 1:
                    # unexpected local loops; choose one and terminate if hits key
                    cands.sort()
                ny2, nx2 = cands[0]
                py, px = cy, cx
                cy, cx = ny2, nx2
                if visited[cy, cx] > 0:
                    break
                visited[cy, cx] = 1
                path.append((cy, cx))
            pix = np.array(path, dtype=np.int32)
            if len(pix) < 2:
                continue
            t0 = _estimate_tangent(pix, at_start=True)
            t1 = _estimate_tangent(pix, at_start=False)
            segs.append(Segment(sid, pix, tuple(pix[0]), tuple(pix[-1]), t0, t1, 1.0))
            sid += 1

    # estimate width from distance transform
    dist = cv2.distanceTransform((vessel_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    for s in segs:
        ys, xs = s.pixels[:, 0], s.pixels[:, 1]
        s.width = float(np.mean(dist[ys, xs]) * 2.0)

    return skel, segs


def segment_confidence(pred: np.ndarray, seg: Segment):
    ys, xs = seg.pixels[:, 0], seg.pixels[:, 1]
    vals = pred[ys, xs]
    a = np.sum(vals == 1)
    v = np.sum(vals == 2)
    if a + v == 0:
        return 0, 0.0
    score = abs(a - v) / (a + v)
    label = 1 if a >= v else 2
    return label, float(score)


def pair_weight(si: Segment, sj: Segment):
    pi = np.array(si.p1, dtype=np.float32)
    pj = np.array(sj.p0, dtype=np.float32)
    link = pj - pi
    d = np.linalg.norm(link) + 1e-6
    u = link / d

    Aij = max(0.0, float(np.dot(si.t1, sj.t0)))
    Lij = max(0.0, float(np.dot(si.t1, u)) * float(np.dot(sj.t0, -u)))
    Tij = np.exp(-abs(si.width - sj.width) / (max(si.width, sj.width, 1.0)))
    Dij = np.exp(-d / 35.0)
    return Aij * Lij * Tij * Dij


def propagate_segments(pred: np.ndarray, segs: List[Segment], max_dist=45.0, iters=2):
    if not segs:
        return pred
    labels: Dict[int, int] = {}
    confs: Dict[int, float] = {}
    for s in segs:
        lab, cf = segment_confidence(pred, s)
        labels[s.sid] = lab
        confs[s.sid] = cf

    edges = []
    for i, si in enumerate(segs):
        for j, sj in enumerate(segs):
            if i == j:
                continue
            d = np.linalg.norm(np.array(si.p1) - np.array(sj.p0))
            if d > max_dist:
                continue
            w = pair_weight(si, sj)
            if w > 0.01:
                edges.append((si.sid, sj.sid, w))

    for _ in range(iters):
        new_labels = labels.copy()
        for sid in labels.keys():
            score_a, score_v = 0.0, 0.0
            base = confs[sid]
            if labels[sid] == 1:
                score_a += base
            elif labels[sid] == 2:
                score_v += base

            for u, v, w in edges:
                if v != sid:
                    continue
                if labels[u] == 1:
                    score_a += w * confs[u]
                elif labels[u] == 2:
                    score_v += w * confs[u]
            if score_a > score_v:
                new_labels[sid] = 1
            elif score_v > score_a:
                new_labels[sid] = 2
        labels = new_labels

    out = pred.copy()
    for s in segs:
        if labels[s.sid] == 0:
            continue
        ys, xs = s.pixels[:, 0], s.pixels[:, 1]
        out[ys, xs] = labels[s.sid]
    return out


def postprocess(pred, mask):
    x = pred.copy()
    x[mask == 0] = 0
    skel, segs = extract_segments(mask)
    x = propagate_segments(x, segs)
    # fill full vessel by nearest skeleton label + local voting
    k = 5
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    ys, xs = np.where(mask > 0)
    for y, x0 in zip(ys, xs):
        if x[y, x0] in (1, 2):
            continue
        win = padded[y:y+k, x0:x0+k]
        vals = win[(win == 1) | (win == 2)]
        if len(vals) > 0:
            x[y, x0] = 1 if np.sum(vals == 1) >= np.sum(vals == 2) else 2
    x[mask == 0] = 0
    return x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(pred_dir.iterdir()):
        if not fp.is_file():
            continue
        pred = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir / fp.name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        out = postprocess(pred, mask)
        cv2.imwrite(str(out_dir / fp.name), out.astype(np.uint8))
