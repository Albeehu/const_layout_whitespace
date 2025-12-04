
#可調整元素大小及位置
# -*- coding: utf-8 -*-
"""
layout_refiner.py
-----------------
Gradient-free layout refiner that *moves/resizes* boxes to seek a
"breathing, high-end whitespace" arrangement.

- Input: boxes = np.array([[x,y,w,h], ...]) in [0,1]
- Output: refined boxes (same shape) + best score
- Objective: aesthetic_whitespace_score (imported from aesthetic_whitespace.py)

This refiner keeps everything inside the page, resolves overlaps softly,
and accepts changes via hill climbing + mild simulated annealing so it
doesn't get stuck too early.

Usage (example):
    from layout_refiner import refine_layout
    best_boxes, best_score = refine_layout(boxes, steps=300, seed=0)

Requirements:
    - numpy
    - aesthetic_whitespace.py available in the same path or PYTHONPATH
"""

import math
import numpy as np

try:
    from aesthetic_whitespace import (
        rasterize_mask_from_boxes,
        aesthetic_whitespace_score,
        grid_snap_boxes, prune_max_items
    )
except Exception as e:
    raise ImportError("Please place aesthetic_whitespace.py alongside this file. Error: %s" % e)


# --------------------- utilities ---------------------

def clamp_boxes_inside(boxes: np.ndarray) -> np.ndarray:
    b = boxes.copy()
    b[:, 0] = np.clip(b[:, 0], 0.0, 1.0)  # x
    b[:, 1] = np.clip(b[:, 1], 0.0, 1.0)  # y
    b[:, 2] = np.clip(b[:, 2], 1e-4, 1.0) # w
    b[:, 3] = np.clip(b[:, 3], 1e-4, 1.0) # h
    # keep inside by shifting if overflow
    b[:, 0] = np.clip(b[:, 0], 0.0, 1.0 - b[:, 2])
    b[:, 1] = np.clip(b[:, 1], 0.0, 1.0 - b[:, 3])
    return b


def overlap_penalty(boxes: np.ndarray) -> float:
    """ Sum of pairwise overlap area normalized by page area. """
    n = boxes.shape[0]
    if n <= 1: return 0.0
    pen = 0.0
    for i in range(n):
        x1,y1,w1,h1 = boxes[i]
        X1a,X2a = x1, x1+w1
        Y1a,Y2a = y1, y1+h1
        for j in range(i+1,n):
            x2,y2,w2,h2 = boxes[j]
            X1b,X2b = x2, x2+w2
            Y1b,Y2b = y2, y2+h2
            ix = max(0.0, min(X2a, X2b) - max(X1a, X1b))
            iy = max(0.0, min(Y2a, Y2b) - max(Y1a, Y1b))
            pen += ix * iy
    return float(pen)  # page area = 1


def score_boxes(boxes: np.ndarray) -> float:
    mask = rasterize_mask_from_boxes(boxes, R=256)
    s = aesthetic_whitespace_score(mask, overlap_pen=overlap_penalty(boxes))
    return float(s)


# ------------------- stochastic ops -------------------

def op_move(boxes: np.ndarray, rng: np.random.Generator, step=0.03) -> np.ndarray:
    b = boxes.copy()
    if b.size == 0: return b
    i = rng.integers(0, b.shape[0])
    dx, dy = rng.normal(0, step, size=2)
    b[i, 0] += dx
    b[i, 1] += dy
    return clamp_boxes_inside(b)


def op_resize(boxes: np.ndarray, rng: np.random.Generator, scale=0.1, keep_aspect=True) -> np.ndarray:
    b = boxes.copy()
    if b.size == 0: return b
    i = rng.integers(0, b.shape[0])
    if keep_aspect:
        f = float(1.0 + rng.normal(0, scale))
        f = np.clip(f, 0.6, 1.4)
        # scale around center
        cx = b[i,0] + b[i,2]/2.0
        cy = b[i,1] + b[i,3]/2.0
        b[i,2] *= f
        b[i,3] *= f
        b[i,0] = cx - b[i,2]/2.0
        b[i,1] = cy - b[i,3]/2.0
    else:
        fx = float(1.0 + rng.normal(0, scale))
        fy = float(1.0 + rng.normal(0, scale))
        fx = np.clip(fx, 0.6, 1.4)
        fy = np.clip(fy, 0.6, 1.4)
        cx = b[i,0] + b[i,2]/2.0
        cy = b[i,1] + b[i,3]/2.0
        b[i,2] *= fx
        b[i,3] *= fy
        b[i,0] = cx - b[i,2]/2.0
        b[i,1] = cy - b[i,3]/2.0
    return clamp_boxes_inside(b)


def op_align_axis(boxes: np.ndarray, rng: np.random.Generator, axis='x', strength=0.5) -> np.ndarray:
    """ Pull a random subset onto a shared x or y (column/row) line. """
    b = boxes.copy()
    n = b.shape[0]
    if n <= 1: return b
    k = int(max(2, rng.integers(n//2, n)))  # random group
    idx = rng.choice(n, size=k, replace=False)
    if axis == 'x':
        # align left edges near their median
        target = np.median(b[idx, 0])
        b[idx, 0] = strength*target + (1-strength)*b[idx, 0]
    else:
        target = np.median(b[idx, 1])
        b[idx, 1] = strength*target + (1-strength)*b[idx, 1]
    return clamp_boxes_inside(b)


def op_cluster(boxes: np.ndarray, rng: np.random.Generator, strength=0.4) -> np.ndarray:
    """ Gently pull all boxes toward their centroid to form a tight block. """
    b = boxes.copy()
    if b.size == 0: return b
    cx = np.mean(b[:,0] + b[:,2]/2.0)
    cy = np.mean(b[:,1] + b[:,3]/2.0)
    b[:,0] = strength*(cx - b[:,2]/2.0) + (1-strength)*b[:,0]
    b[:,1] = strength*(cy - b[:,3]/2.0) + (1-strength)*b[:,1]
    return clamp_boxes_inside(b)


def op_margins_inflate(boxes: np.ndarray, rng: np.random.Generator, pad=0.04) -> np.ndarray:
    """ Scale the whole content bbox inward to create larger outer margins. """
    b = boxes.copy()
    if b.size == 0: return b
    # content bbox
    x1 = np.min(b[:,0]); y1 = np.min(b[:,1])
    x2 = np.max(b[:,0]+b[:,2]); y2 = np.max(b[:,1]+b[:,3])
    # shrink toward the bbox center by pad fraction
    cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
    sx = 1.0 - pad; sy = 1.0 - pad
    # move each box center toward content center, shrink box size a touch
    for i in range(b.shape[0]):
        bx = b[i,0] + b[i,2]/2.0
        by = b[i,1] + b[i,3]/2.0
        bx = cx + (bx - cx)*sx
        by = cy + (by - cy)*sy
        b[i,2] *= sx
        b[i,3] *= sy
        b[i,0] = bx - b[i,2]/2.0
        b[i,1] = by - b[i,3]/2.0
    return clamp_boxes_inside(b)


def soft_resolve_overlaps(boxes: np.ndarray, iters=50, repel=0.01) -> np.ndarray:
    """ Small repulsive nudges to reduce overlaps without wrecking layout. """
    b = boxes.copy()
    for _ in range(iters):
        moved = False
        n = b.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, wi, hi = b[i]
                xj, yj, wj, hj = b[j]
                ix = min(xi+wi, xj+wj) - max(xi, xj)
                iy = min(yi+hi, yj+hj) - max(yi, yj)
                if ix > 0 and iy > 0:
                    # push apart along the smaller overlap axis
                    if ix < iy:
                        dx = repel if (xi+wi/2) < (xj+wj/2) else -repel
                        b[i,0] -= dx; b[j,0] += dx
                    else:
                        dy = repel if (yi+hi/2) < (yj+hj/2) else -repel
                        b[i,1] -= dy; b[j,1] += dy
                    moved = True
        if not moved:
            break
        b = clamp_boxes_inside(b)
    return b


OPS = [
    lambda b, r: op_move(b, r, step=0.03),
    lambda b, r: op_resize(b, r, scale=0.12, keep_aspect=True),
    lambda b, r: op_align_axis(b, r, axis='x', strength=0.6),
    lambda b, r: op_align_axis(b, r, axis='y', strength=0.6),
    lambda b, r: op_cluster(b, r, strength=0.35),
    lambda b, r: op_margins_inflate(b, r, pad=0.04),
]


def refine_layout(
    boxes: np.ndarray,
    steps: int = 300,
    seed: int = 0,
    snap_step: float = 0.04,
    prune_items: int = None,
    anneal_start: float = 0.15,
    anneal_end: float = 0.02
):
    """
    Run a stochastic search that *moves/resizes* boxes while scoring with
    aesthetic_whitespace_score. Returns (best_boxes, best_score).

    - snap_step: after each accepted move, snap to coarse grid for cleaner whitespace
    - prune_items: if set (e.g., 3), will prune to <= N items once at the start
    - anneal_*: acceptance temperature schedule
    """
    rng = np.random.default_rng(seed)
    cur = boxes.copy()

    if prune_items is not None and prune_items > 0:
        cur = prune_max_items(cur, max_items=prune_items, min_area=0.012)

    cur = clamp_boxes_inside(cur)
    cur = soft_resolve_overlaps(cur, iters=40, repel=0.01)

    cur_score = score_boxes(cur)
    best = cur.copy()
    best_score = cur_score

    def temperature(t):
        return anneal_start * (anneal_end/anneal_start) ** (t / max(1, steps-1))

    for t in range(steps):
        cand = cur.copy()
        # choose 1~2 random ops
        k = 2 if rng.random() < 0.35 else 1
        for _ in range(k):
            op = OPS[rng.integers(0, len(OPS))]
            cand = op(cand, rng)

        cand = soft_resolve_overlaps(cand, iters=20, repel=0.008)
        cand = clamp_boxes_inside(cand)
        cand = grid_snap_boxes(cand, step=snap_step)

        s = score_boxes(cand)
        if s >= cur_score:
            cur, cur_score = cand, s
        else:
            # simulated annealing acceptance
            T = temperature(t)
            accept = math.exp((s - cur_score) / max(1e-6, T))
            if rng.random() < accept:
                cur, cur_score = cand, s

        if cur_score > best_score:
            best, best_score = cur.copy(), cur_score

    return best, float(best_score)


if __name__ == "__main__":
    # Quick demo with three boxes
    init = np.array([
        [0.10, 0.12, 0.25, 0.08],   # title
        [0.55, 0.20, 0.30, 0.18],   # hero
        [0.15, 0.55, 0.32, 0.20],   # body
    ], dtype=np.float32)

    b0 = init.copy()
    s0 = score_boxes(b0)
    print("Initial score:", round(s0, 4))

    best, s_best = refine_layout(b0, steps=250, seed=3, snap_step=0.04, prune_items=3)
    print("Refined score:", round(s_best, 4))
    print("Refined boxes:\n", best)
