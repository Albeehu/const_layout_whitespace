#不固定四周的評分
# -*- coding: utf-8 -*-
"""
aesthetic_whitespace.py
---------------------------------
Drop-in helpers to encourage "breathing", high-end minimal layouts via
mask-based metrics (no hard templates). Use these with your existing pipeline.

Core ideas scored (all 0..1, higher is better):
- breathing_margin_score: thick & uniform outer margins
- compact_block_score: small, compact content bounding region
- largest_connected_whitespace: one big contiguous white region
- white_corridor_score: a clear "air corridor" across the page

Then combine them in aesthetic_whitespace_score. You can blend your
existing signals (WSR, margin consistency, etc.) using the parameters.

This file is self-contained and *only* relies on numpy.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple


def _content_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return (x1, y1, x2, y2)


def breathing_margin_score(mask: np.ndarray, target: float = 0.12) -> float:
    H, W = mask.shape[:2]
    if mask.sum() == 0:
        mt = mb = ml = mr = 1.0
    else:
        row_sum = mask.sum(axis=1) > 0
        col_sum = mask.sum(axis=0) > 0
        top = int(np.argmax(row_sum)) if row_sum.any() else H
        bottom = H - int(np.argmax(row_sum[::-1])) - 1 if row_sum.any() else -1
        left = int(np.argmax(col_sum)) if col_sum.any() else W
        right = W - int(np.argmax(col_sum[::-1])) - 1 if col_sum.any() else -1

        mt = top / float(H)
        mb = (H - bottom - 1) / float(H)
        ml = left / float(W)
        mr = (W - right - 1) / float(W)

    margins = np.array([mt, mb, ml, mr], dtype=np.float32)
    thick = np.clip((margins - target) / (0.22 - target + 1e-6), 0.0, 1.0).mean()
    uniform = 1.0 - np.clip(margins.std() / 0.06, 0.0, 1.0)
    return float(0.6 * thick + 0.4 * uniform)


def compact_block_score(mask: np.ndarray, bbox_target_lo: float = 0.20, bbox_target_hi: float = 0.35) -> float:
    H, W = mask.shape[:2]
    bbox = _content_bbox(mask)
    if bbox is None:
        return 0.7
    x1, y1, x2, y2 = bbox
    bbox_area = (y2 - y1) * (x2 - x1)
    if bbox_area <= 0:
        return 0.0
    page = H * W
    frac = bbox_area / float(page)

    if frac <= bbox_target_lo:
        area_score = 1.0
    elif frac >= bbox_target_hi:
        area_score = 0.0
    else:
        area_score = 1.0 - (frac - bbox_target_lo) / (bbox_target_hi - bbox_target_lo + 1e-6)

    fill = mask[y1:y2, x1:x2].mean()
    compact = np.clip((fill - 0.5) / 0.5, 0.0, 1.0)
    return float(0.55 * area_score + 0.45 * compact)


def largest_connected_whitespace(mask: np.ndarray) -> float:
    white = (mask == 0).astype(np.uint8)
    H, W = white.shape
    visited = np.zeros_like(white, dtype=bool)
    max_area = 0

    for y in range(H):
        for x in range(W):
            if white[y, x] and not visited[y, x]:
                area = 0
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    if cy > 0 and white[cy-1, cx] and not visited[cy-1, cx]:
                        visited[cy-1, cx] = True; stack.append((cy-1, cx))
                    if cy+1 < H and white[cy+1, cx] and not visited[cy+1, cx]:
                        visited[cy+1, cx] = True; stack.append((cy+1, cx))
                    if cx > 0 and white[cy, cx-1] and not visited[cy, cx-1]:
                        visited[cy, cx-1] = True; stack.append((cy, cx-1))
                    if cx+1 < W and white[cy, cx+1] and not visited[cy, cx+1]:
                        visited[cy, cx+1] = True; stack.append((cy, cx+1))
                if area > max_area:
                    max_area = area

    return float(max_area / (H * W + 1e-6))


def white_corridor_score(mask: np.ndarray, min_width_frac: float = 0.06) -> float:
    H, W = mask.shape[:2]
    d = max(1, int(0.01 * max(H, W)))
    m = mask.copy()
    for _ in range(d):
        m = np.maximum.reduce([
            m,
            np.roll(m, 1, axis=0), np.roll(m, -1, axis=0),
            np.roll(m, 1, axis=1), np.roll(m, -1, axis=1),
        ])
    white = (m == 0).astype(np.uint8)

    def _has_vertical_corridor():
        starts = np.where(white[0] > 0)[0]
        if starts.size == 0:
            return False
        visited = np.zeros_like(white, dtype=bool)
        from collections import deque
        dq = deque([(0, int(starts[starts.size // 2]))])
        visited[0, dq[0][1]] = True
        while dq:
            y, x = dq.popleft()
            if y == H - 1:
                return True
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < H and 0 <= nx < W and white[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    dq.append((ny, nx))
        return False

    def _has_horizontal_corridor():
        starts = np.where(white[:, 0] > 0)[0]
        if starts.size == 0:
            return False
        visited = np.zeros_like(white, dtype=bool)
        from collections import deque
        dq = deque([(int(starts[starts.size // 2]), 0)])
        visited[dq[0][0], 0] = True
        while dq:
            y, x = dq.popleft()
            if x == W - 1:
                return True
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < H and 0 <= nx < W and white[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    dq.append((ny, nx))
        return False

    corridor = _has_vertical_corridor() or _has_horizontal_corridor()

    min_run = np.inf
    for y in range(H):
        run = 0
        for x in range(W):
            if white[y, x]:
                run += 1
            else:
                if run > 0:
                    min_run = min(min_run, run); run = 0
        if run > 0:
            min_run = min(min_run, run)
    for x in range(W):
        run = 0
        for y in range(H):
            if white[y, x]:
                run += 1
            else:
                if run > 0:
                    min_run = min(min_run, run); run = 0
        if run > 0:
            min_run = min(min_run, run)

    if not np.isfinite(min_run):
        min_run = 0.0

    width_frac = min_run / float(max(H, W))
    if corridor:
        return float(np.clip((width_frac - min_width_frac) / (0.12 - min_width_frac + 1e-6), 0.0, 1.0))
    else:
        return 0.0


def aesthetic_whitespace_score(
    mask: np.ndarray,
    wsr: Optional[float] = None,
    margin_consistency: Optional[float] = None,
    whitespace_conc: Optional[float] = None,
    overlap_pen: float = 0.0,
    coverage_floor: float = 0.14,
    weights: Optional[dict] = None,
) -> float:
    white = (mask == 0).astype(np.uint8)
    if wsr is None:
        wsr = float(white.mean())

    coverage = 1.0 - wsr
    coverage_gate = 0.0 if coverage < coverage_floor else (coverage - coverage_floor) / (1.0 - coverage_floor + 1e-6)

    bm = breathing_margin_score(mask)
    cb = compact_block_score(mask)
    lcc = largest_connected_whitespace(mask)
    wcorr = white_corridor_score(mask)

    margin_consistency = 0.5 if margin_consistency is None else float(margin_consistency)
    whitespace_conc = 0.5 if whitespace_conc is None else float(whitespace_conc)

    if weights is None:
        weights = {"bm":0.22, "cb":0.22, "lcc":0.18, "wcorr":0.18, "wsr":0.08, "mcs":0.06, "wsc":0.06}

    alpha = 0.55
    base = (
        weights["bm"]*bm +
        weights["cb"]*cb +
        weights["lcc"]*lcc +
        weights["wcorr"]*wcorr +
        weights["wsr"]*(wsr**alpha) +
        weights["mcs"]*margin_consistency +
        weights["wsc"]*whitespace_conc
    ) / (sum(weights.values()) + 1e-6)

    score = coverage_gate * base - 0.6 * max(0.0, overlap_pen)
    return float(np.clip(score, 0.0, 1.0))


def grid_snap_boxes(boxes: np.ndarray, step: float = 0.04) -> np.ndarray:
    def _snap(v: np.ndarray) -> np.ndarray:
        return np.round(v / step) * step
    b = boxes.copy()
    if b.size == 0:
        return b
    b[:, 0] = np.clip(_snap(b[:, 0]), 0.0, 1.0)
    b[:, 1] = np.clip(_snap(b[:, 1]), 0.0, 1.0)
    b[:, 2] = np.clip(_snap(b[:, 2]), step, 1.0)
    b[:, 3] = np.clip(_snap(b[:, 3]), step, 1.0)
    b[:, 0] = np.clip(b[:, 0], 0.0, 1.0 - b[:, 2])
    b[:, 1] = np.clip(b[:, 1], 0.0, 1.0 - b[:, 3])
    return b


def prune_max_items(
    boxes: np.ndarray,
    scores: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    max_items: int = 3,
    min_area: float = 0.012
) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    areas = boxes[:, 2] * boxes[:, 3]
    keep = areas >= min_area
    b = boxes[keep]
    if b.shape[0] <= max_items:
        return b
    if scores is None:
        scores = np.ones(b.shape[0], dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if labels is None:
        pri = np.ones_like(scores)
    else:
        pr_map = {"title":1.0, "text":0.9, "table":0.8, "list":0.7, "figure":0.5, "decoration":0.3, "unknown":0.4, "view":0.5}
        pri = np.array([pr_map.get(l, 0.5) for l in labels], dtype=np.float32)
    order = np.argsort(-(pri * scores))
    return b[order[:max_items]]


def rasterize_mask_from_boxes(boxes: np.ndarray, R: int = 256) -> np.ndarray:
    mask = np.zeros((R, R), dtype=np.uint8)
    H = W = R
    for x, y, w, h in boxes:
        x1 = int(np.clip(x * W, 0, W - 1))
        y1 = int(np.clip(y * H, 0, H - 1))
        x2 = int(np.clip((x + w) * W, 0, W))
        y2 = int(np.clip((y + h) * H, 0, H))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


if __name__ == "__main__":
    boxes = np.array([[0.65, 0.65, 0.22, 0.18]], dtype=np.float32)
    mask = rasterize_mask_from_boxes(boxes, R=256)
    s = aesthetic_whitespace_score(mask)
    print("Sanity score:", round(s, 4))
