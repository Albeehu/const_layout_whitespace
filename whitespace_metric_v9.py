#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
whitespace_metric_v9_coarse.py

For generated_layouts.pkl produced by layout generation / inference scripts:
    results = [
        (boxes, labels),   # boxes usually in cx, cy, w, h
        ...
    ]

Main metrics:
- WR:  Whitespace Ratio = whitespace area / canvas area
- LWR: Largest Whitespace Ratio = largest connected whitespace area / canvas area
- D_conn = LWR / WR
- S_frame / S_side / S_tb / S_corner: style-position scores from layout margins
- S_style: selected style score controlled by --target_style
- Q_overlap = exp(-beta_overlap * overlap_ratio)
- DWS = D_conn * S_style * Q_overlap
- DWS_max = D_conn * S_max * Q_overlap

New in this version:
- Optional coarse whitespace preprocessing:
  1) content dilation: expands occupied regions so tiny gaps are not counted as whitespace
  2) block downsampling: evaluates whitespace at a coarser visual scale

Recommended start for 128x128 layouts:
    --use_coarse_ws --dilation_px 2 --block_size 4 --block_occ_ratio 0.15

Why this matters:
- If most background whitespace is connected, LWR / WR often becomes close to 1.
- Coarse preprocessing reduces the influence of tiny cracks/gaps between elements.
- Bounding boxes are treated as filled content, so outline/wireframe elements will not be
  misread as internal whitespace.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_dilation


# -----------------------------------------------------------------------------
# Box format conversion
# -----------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def xywh_topleft_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x2 = x + w
    y2 = y + h
    return np.stack([x, y, x2, y2], axis=-1)


def convert_boxes_to_xyxy(boxes: np.ndarray, box_format: str) -> np.ndarray:
    if box_format == "cxcywh":
        return cxcywh_to_xyxy(boxes)
    if box_format == "xywh":
        return xywh_topleft_to_xyxy(boxes)
    if box_format == "xyxy":
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    raise ValueError(f"Unsupported box_format: {box_format}")


# -----------------------------------------------------------------------------
# Mask construction and coarse whitespace preprocessing
# -----------------------------------------------------------------------------

def filter_boxes_by_labels(
    boxes: np.ndarray,
    labels: np.ndarray | None,
    ignore_labels: List[int],
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if labels is None:
        return boxes
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != len(boxes):
        raise ValueError(f"labels length {len(labels)} != boxes length {len(boxes)}")
    if not ignore_labels:
        return boxes
    keep = ~np.isin(labels, np.asarray(ignore_labels))
    return boxes[keep]


def layout_to_mask(
    boxes_xyxy: np.ndarray,
    height: int = 128,
    width: int = 128,
    large_thresh: float = 0.95,
) -> np.ndarray:
    """Convert boxes to an occupied/content mask.

    Returns:
        occ_mask: np.uint8 array, shape (height, width)
                  1 = occupied/content, 0 = whitespace

    Important:
        Each bounding box is filled. This is intentional for layout-level whitespace
        evaluation. If visual elements are outline-only, their whole bounding-box region
        is still treated as occupied layout space.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return mask

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)

    for (x1, y1, x2, y2) in boxes_xyxy:
        x1 = float(np.clip(x1, 0.0, 1.0))
        y1 = float(np.clip(y1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0))
        y2 = float(np.clip(y2, 0.0, 1.0))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh

        # Ignore almost-full-canvas boxes, often background containers.
        if area >= large_thresh:
            continue

        x1p = int(x1 * width)
        x2p = int(np.ceil(x2 * width))
        y1p = int(y1 * height)
        y2p = int(np.ceil(y2 * height))

        x1p = int(np.clip(x1p, 0, width))
        x2p = int(np.clip(x2p, 0, width))
        y1p = int(np.clip(y1p, 0, height))
        y2p = int(np.clip(y2p, 0, height))

        if x2p <= x1p or y2p <= y1p:
            continue

        mask[y1p:y2p, x1p:x2p] = 1

    return mask


def preprocess_whitespace_mask(
    occ_mask: np.ndarray,
    use_coarse_ws: bool = False,
    dilation_px: int = 2,
    block_size: int = 4,
    block_occ_ratio: float = 0.15,
) -> np.ndarray:
    """Create a whitespace mask from an occupied mask.

    Args:
        occ_mask:
            1 = occupied/content, 0 = whitespace.
        use_coarse_ws:
            If False, returns the original pixel-level whitespace mask.
            If True, applies content dilation and block downsampling.
        dilation_px:
            Radius-like size for content dilation. Larger values remove more tiny gaps.
            For 128x128 layouts, start with 1~3.
        block_size:
            Coarse block size. For 128x128 layouts, start with 4 or 8.
        block_occ_ratio:
            If a block contains at least this ratio of content pixels, the whole block is
            treated as occupied. Lower values are stricter and remove more tiny gaps.

    Returns:
        whitespace_mask: np.uint8 array, 1 = whitespace, 0 = content.
            If use_coarse_ws=True, the returned mask is lower-resolution.
    """
    occ = np.asarray(occ_mask).astype(bool)

    if not use_coarse_ws:
        return (~occ).astype(np.uint8)

    if dilation_px < 0:
        raise ValueError("dilation_px must be >= 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if not (0.0 <= block_occ_ratio <= 1.0):
        raise ValueError("block_occ_ratio must be in [0, 1]")

    # 1) Expand content to absorb small gaps between elements.
    if dilation_px > 0:
        k = 2 * int(dilation_px) + 1
        structure = np.ones((k, k), dtype=bool)
        occ = binary_dilation(occ, structure=structure)

    h, w = occ.shape
    new_h = h // block_size * block_size
    new_w = w // block_size * block_size

    if new_h <= 0 or new_w <= 0:
        raise ValueError(
            f"block_size={block_size} is too large for mask shape {(h, w)}"
        )

    # 2) Crop to exact multiple of block size.
    occ = occ[:new_h, :new_w]

    # 3) Downsample by block occupancy ratio.
    blocks = occ.reshape(
        new_h // block_size,
        block_size,
        new_w // block_size,
        block_size,
    )
    block_occ = blocks.mean(axis=(1, 3))

    # If enough content appears in a block, treat the whole block as content.
    coarse_occ = block_occ >= block_occ_ratio
    coarse_whitespace = ~coarse_occ

    return coarse_whitespace.astype(np.uint8)


# -----------------------------------------------------------------------------
# Whitespace metrics
# -----------------------------------------------------------------------------

def whitespace_metrics_from_mask(
    whitespace_mask: np.ndarray,
    alpha: float = 0.1,
    connectivity: int = 4,
) -> Dict[str, float]:
    """Compute WR, LWR, D_conn and fragmentation penalty from a whitespace mask.

    whitespace_mask:
        1 = whitespace, 0 = content.

    connectivity:
        4 or 8. For layout whitespace, 4-connectivity is usually stricter.
    """
    whitespace_mask = np.asarray(whitespace_mask).astype(bool)
    h, w = whitespace_mask.shape
    total_pixels = float(h * w)

    wr = float(whitespace_mask.sum()) / total_pixels
    if wr < 1e-8:
        return dict(
            WR=wr, LWR=0.0, DWS=0.0, DWS_max=0.0, num_cc=0.0,
            cx=0.5, cy=0.5, D_conn=0.0, frag_penalty=0.0,
            metric_h=float(h), metric_w=float(w),
        )

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    elif connectivity == 4:
        structure = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.uint8)
    else:
        raise ValueError("connectivity must be 4 or 8")

    labeled, num_cc = cc_label(whitespace_mask, structure=structure)
    if num_cc == 0:
        return dict(
            WR=wr, LWR=0.0, DWS=0.0, DWS_max=0.0, num_cc=0.0,
            cx=0.5, cy=0.5, D_conn=0.0, frag_penalty=0.0,
            metric_h=float(h), metric_w=float(w),
        )

    areas = np.bincount(labeled.ravel())[1:]
    largest_area = float(areas.max())
    lwr = largest_area / total_pixels

    ys, xs = np.nonzero(whitespace_mask)
    if len(xs) == 0:
        cx = cy = 0.5
    else:
        cx = float(xs.mean() / float(w))
        cy = float(ys.mean() / float(h))

    d_conn = lwr / (wr + 1e-8) if wr > 1e-8 else 0.0
    frag_penalty = 1.0 / (1.0 + alpha * max(0.0, float(num_cc) - 1.0))

    return dict(
        WR=wr,
        LWR=lwr,
        DWS=0.0,
        DWS_max=0.0,
        num_cc=float(num_cc),
        cx=cx,
        cy=cy,
        D_conn=float(d_conn),
        frag_penalty=float(frag_penalty),
        metric_h=float(h),
        metric_w=float(w),
    )


# -----------------------------------------------------------------------------
# Position / style metrics
# -----------------------------------------------------------------------------

def content_margins_from_boxes(boxes_xyxy: np.ndarray) -> Tuple[float, float, float, float]:
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)

    if boxes_xyxy.size == 0:
        return 1.0, 1.0, 1.0, 1.0

    x1 = np.clip(boxes_xyxy[:, 0], 0.0, 1.0)
    y1 = np.clip(boxes_xyxy[:, 1], 0.0, 1.0)
    x2 = np.clip(boxes_xyxy[:, 2], 0.0, 1.0)
    y2 = np.clip(boxes_xyxy[:, 3], 0.0, 1.0)

    x1_min = float(x1.min())
    y1_min = float(y1.min())
    x2_max = float(x2.max())
    y2_max = float(y2.max())

    L = float(np.clip(x1_min, 0.0, 1.0))
    R = float(np.clip(1.0 - x2_max, 0.0, 1.0))
    T = float(np.clip(y1_min, 0.0, 1.0))
    B = float(np.clip(1.0 - y2_max, 0.0, 1.0))
    return L, R, T, B


def normalize_target_style(target_style: str) -> str:
    target_style = str(target_style).lower().strip()
    alias = {
        "right": "side",
        "top": "tb",
        "hybrid": "corner",
    }
    target_style = alias.get(target_style, target_style)
    allowed = {"mean", "max", "frame", "side", "tb", "corner"}
    if target_style not in allowed:
        raise ValueError(
            f"Unsupported target_style={target_style}. "
            f"Use one of {sorted(allowed)} or aliases: right/top/hybrid."
        )
    return target_style


def style_scores_from_margins(
    L: float,
    R: float,
    T: float,
    B: float,
    target_style: str = "mean",
) -> Dict[str, float]:
    margins = np.array([L, R, T, B], dtype=np.float32)
    mean_m = float(margins.mean())
    std_m = float(margins.std())

    # Frame: balanced margins plus tolerance for vertical offset compositions.
    S_margin = mean_m - 0.5 * std_m
    S_vertical = abs(T - B)

    content_top = T
    content_bottom = 1.0 - B
    cy_content = (content_top + content_bottom) / 2.0

    vertical_tolerance = 0.25
    P_extreme = max(0.0, abs(cy_content - 0.5) - vertical_tolerance)

    S_frame = (0.6 * S_margin) + (0.4 * S_vertical) - (0.3 * P_extreme)

    horiz = np.array([L, R], dtype=np.float32)
    vert = np.array([T, B], dtype=np.float32)

    h_max, h_min, h_mean = float(horiz.max()), float(horiz.min()), float(horiz.mean())
    v_max, v_min, v_mean = float(vert.max()), float(vert.min()), float(vert.mean())

    # Side whitespace style: reward left/right large whitespace and asymmetry.
    h_diff = h_max - h_min
    S_side_raw = h_max + 0.8 * h_diff + 0.2 * v_mean

    h_edge_threshold = 0.05
    h_edge_penalty = (
        1.0
        if h_min >= h_edge_threshold
        else 0.7 + 0.3 * (h_min / h_edge_threshold)
    )
    S_side = S_side_raw * h_edge_penalty

    # Top-bottom whitespace style: reward vertical large whitespace and asymmetry.
    v_diff = v_max - v_min
    S_tb_raw = v_max + 0.8 * v_diff + 0.2 * h_mean

    v_edge_threshold = 0.05
    v_edge_penalty = (
        1.0
        if v_min >= v_edge_threshold
        else 0.7 + 0.3 * (v_min / v_edge_threshold)
    )
    S_tb = S_tb_raw * v_edge_penalty

    S_frame = float(np.clip(S_frame, 0.0, 1.0))
    S_side = float(np.clip(S_side, 0.0, 1.0))
    S_tb = float(np.clip(S_tb, 0.0, 1.0))

    # Corner / hybrid style combines side and top-bottom whitespace.
    S_corner = float(np.sqrt(max(0.0, S_side * S_tb)))

    style_map = {
        "frame": S_frame,
        "side": S_side,
        "tb": S_tb,
        "corner": S_corner,
    }
    dominant_style = max(style_map, key=style_map.get)
    S_max = float(max(style_map.values()))

    target_style_norm = normalize_target_style(target_style)
    if target_style_norm == "mean":
        S_style = float((S_frame + S_side + S_tb + S_corner) / 4.0)
    elif target_style_norm == "max":
        S_style = S_max
    else:
        S_style = float(style_map[target_style_norm])

    return dict(
        S_frame=S_frame,
        S_side=S_side,
        S_tb=S_tb,
        S_corner=S_corner,
        S_style=S_style,
        S_max=S_max,
        S_pos=S_max,
        dominant_style=dominant_style,
        target_style=target_style_norm,
    )


# -----------------------------------------------------------------------------
# Overlap penalty
# -----------------------------------------------------------------------------

def compute_overlap_ratio_xyxy(
    boxes_xyxy: np.ndarray,
    labels: np.ndarray | None = None,
    eps: float = 1e-8,
) -> float:
    """Compute label-aware weighted overlap ratio.

    overlap_ratio = weighted_pairwise_overlap_area / foreground_area

    Label convention used by the inference code:
    - 0 = SVG
    - 1 = text
    - 2 = image
    - 5 = face

    Rules:
    - Face-image overlap is allowed and not penalized.
    - Text overlap and face-vs-non-image overlap are penalized more strongly.
    - Face area is excluded from foreground_area denominator.
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    n = len(boxes_xyxy)
    if n <= 1:
        return 0.0

    if labels is None:
        labels_arr = np.zeros((n,), dtype=np.int64)
    else:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(labels_arr) != n:
            raise ValueError(f"labels length {len(labels_arr)} != boxes length {n}")

    b = boxes_xyxy.copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, 1.0)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, 1.0)

    widths = np.maximum(0.0, b[:, 2] - b[:, 0])
    heights = np.maximum(0.0, b[:, 3] - b[:, 1])
    areas = widths * heights

    fg_mask = labels_arr != 5
    fg_area = float(np.sum(areas[fg_mask]))
    if fg_area <= eps:
        return 0.0

    weighted_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            li = int(labels_arr[i])
            lj = int(labels_arr[j])

            # Face inside image is expected, so do not penalize it.
            if {li, lj} == {2, 5}:
                continue

            ix1 = max(float(b[i, 0]), float(b[j, 0]))
            iy1 = max(float(b[i, 1]), float(b[j, 1]))
            ix2 = min(float(b[i, 2]), float(b[j, 2]))
            iy2 = min(float(b[i, 3]), float(b[j, 3]))
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0.0:
                continue

            if li == 1 or lj == 1:
                weight = 2.0
            elif li == 5 or lj == 5:
                weight = 2.0
            else:
                weight = 1.0

            weighted_overlap += weight * inter

    return float(np.clip(weighted_overlap / max(fg_area, eps), 0.0, 1.0))


# -----------------------------------------------------------------------------
# Main quality function
# -----------------------------------------------------------------------------

def whitespace_quality(
    boxes_xyxy: np.ndarray,
    height: int = 128,
    width: int = 128,
    alpha: float = 0.1,
    large_thresh: float = 0.95,
    boxes_all_xyxy: np.ndarray | None = None,
    labels_all: np.ndarray | None = None,
    beta_overlap: float = 5.0,
    target_style: str = "mean",
    use_coarse_ws: bool = False,
    dilation_px: int = 2,
    block_size: int = 4,
    block_occ_ratio: float = 0.15,
    connectivity: int = 4,
) -> Dict[str, float]:
    """Compute whitespace/style/overlap metrics for one layout."""
    occ_mask = layout_to_mask(
        boxes_xyxy,
        height=height,
        width=width,
        large_thresh=large_thresh,
    )

    whitespace_mask = preprocess_whitespace_mask(
        occ_mask,
        use_coarse_ws=use_coarse_ws,
        dilation_px=dilation_px,
        block_size=block_size,
        block_occ_ratio=block_occ_ratio,
    )

    m = whitespace_metrics_from_mask(
        whitespace_mask,
        alpha=alpha,
        connectivity=connectivity,
    )

    L, R, T, B = content_margins_from_boxes(boxes_xyxy)
    style = style_scores_from_margins(L, R, T, B, target_style=target_style)

    overlap_boxes = boxes_all_xyxy if boxes_all_xyxy is not None else boxes_xyxy
    overlap_ratio = compute_overlap_ratio_xyxy(overlap_boxes, labels=labels_all)
    Q_overlap = float(np.exp(-float(beta_overlap) * overlap_ratio))

    # Final score definition.
    m["DWS"] = float(m["D_conn"] * style["S_style"] * Q_overlap)
    m["DWS_max"] = float(m["D_conn"] * style["S_max"] * Q_overlap)

    m.update(dict(
        L=L,
        R=R,
        T=T,
        B=B,
        S_frame=style["S_frame"],
        S_side=style["S_side"],
        S_tb=style["S_tb"],
        S_corner=style["S_corner"],
        S_style=style["S_style"],
        S_max=style["S_max"],
        S_pos=style["S_pos"],
        dominant_style=style["dominant_style"],
        target_style=style["target_style"],
        overlap_ratio=overlap_ratio,
        Q_overlap=Q_overlap,
        beta_overlap=float(beta_overlap),
        use_coarse_ws=float(bool(use_coarse_ws)),
        dilation_px=float(dilation_px),
        block_size=float(block_size),
        block_occ_ratio=float(block_occ_ratio),
        connectivity=float(connectivity),
    ))
    return m


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_generated_layouts(
    path: Path,
    box_format: str,
    ignore_labels: List[int],
) -> List[Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]]:
    """Load layouts.

    Returns a list of:
        (boxes_metric_xyxy, labels_metric, boxes_all_xyxy, labels_all)

    boxes_metric_xyxy applies ignore_labels and is used for whitespace masks/margins.
    boxes_all_xyxy keeps all labels and is used for label-aware overlap penalties.
    """
    with path.open("rb") as f:
        data = pickle.load(f)

    layouts: List[Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]] = []

    if not isinstance(data, list):
        raise ValueError("Expected generated_layouts.pkl to contain a list.")

    for item in data:
        labels = None
        boxes = None

        if isinstance(item, tuple):
            if len(item) >= 2:
                boxes = np.asarray(item[0], dtype=np.float32).reshape(-1, 4)
                labels = np.asarray(item[1]).reshape(-1)
            elif len(item) == 1:
                boxes = np.asarray(item[0], dtype=np.float32).reshape(-1, 4)
        elif isinstance(item, dict):
            if "boxes" in item:
                boxes = np.asarray(item["boxes"], dtype=np.float32).reshape(-1, 4)
            elif "bboxes" in item:
                boxes = np.asarray(item["bboxes"], dtype=np.float32).reshape(-1, 4)
            if "labels" in item:
                labels = np.asarray(item["labels"]).reshape(-1)
        else:
            boxes = np.asarray(item, dtype=np.float32).reshape(-1, 4)

        if boxes is None:
            raise ValueError("Could not find boxes in one layout item.")

        boxes_all_xyxy = convert_boxes_to_xyxy(boxes, box_format)
        labels_all = labels.copy() if labels is not None else None

        if labels is not None and ignore_labels:
            keep = ~np.isin(labels, np.asarray(ignore_labels))
            boxes_metric = boxes[keep]
            labels_metric = labels[keep]
        else:
            boxes_metric = boxes
            labels_metric = labels

        boxes_metric_xyxy = convert_boxes_to_xyxy(boxes_metric, box_format)
        layouts.append((boxes_metric_xyxy, labels_metric, boxes_all_xyxy, labels_all))

    return layouts


# -----------------------------------------------------------------------------
# Summary and output
# -----------------------------------------------------------------------------

SUMMARY_KEYS = [
    "WR", "LWR", "DWS", "DWS_max", "num_cc",
    "L", "R", "T", "B",
    "cx", "cy",
    "D_conn", "frag_penalty",
    "S_frame", "S_side", "S_tb", "S_corner", "S_style", "S_max", "S_pos",
    "overlap_ratio", "Q_overlap",
    "metric_h", "metric_w",
]


def mean_std(arr: List[float]) -> Tuple[float, float]:
    if len(arr) == 0:
        return 0.0, 0.0
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std())


def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {
        "num_layouts": int(len(metrics_list)),
    }

    for key in SUMMARY_KEYS:
        values = [float(m.get(key, 0.0)) for m in metrics_list]
        mu, sd = mean_std(values)
        summary[f"mean_{key}"] = mu
        summary[f"std_{key}"] = sd

    counts = {"frame": 0, "side": 0, "tb": 0, "corner": 0}
    for m in metrics_list:
        ds = m.get("dominant_style", "frame")
        if ds in counts:
            counts[ds] += 1

    n = max(1, len(metrics_list))
    for k, v in counts.items():
        summary[f"count_{k}"] = int(v)
        summary[f"ratio_{k}"] = float(v / n)

    # Store common run configuration from first metric row for traceability.
    if metrics_list:
        first = metrics_list[0]
        for k in [
            "use_coarse_ws", "dilation_px", "block_size", "block_occ_ratio",
            "connectivity", "beta_overlap",
        ]:
            if k in first:
                summary[k] = float(first[k])
        if "target_style" in first:
            summary["target_style"] = str(first["target_style"])

    return summary


def diff_summary(curr: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    diff = {}
    numeric_keys = sorted(set(curr.keys()) & set(base.keys()))
    for k in numeric_keys:
        if isinstance(curr[k], (int, float)) and isinstance(base[k], (int, float)):
            diff[k] = float(curr[k]) - float(base[k])
    return diff


def format_summary_block(name: str, summary: Dict[str, float]) -> str:
    lines = [f"=== {name} ==="]
    lines.append(f"num_layouts={int(summary.get('num_layouts', 0))}")

    config_keys = [
        "target_style", "use_coarse_ws", "dilation_px", "block_size",
        "block_occ_ratio", "connectivity", "beta_overlap",
    ]
    for k in config_keys:
        if k in summary:
            lines.append(f"{k}={summary[k]}")

    core = [
        "mean_WR", "mean_LWR", "mean_D_conn", "mean_DWS", "mean_DWS_max",
        "mean_num_cc", "mean_frag_penalty",
        "mean_S_frame", "mean_S_side", "mean_S_tb", "mean_S_corner", "mean_S_style", "mean_S_max",
        "mean_overlap_ratio", "mean_Q_overlap",
        "mean_L", "mean_R", "mean_T", "mean_B",
        "mean_metric_h", "mean_metric_w",
        "ratio_frame", "ratio_side", "ratio_tb", "ratio_corner",
    ]
    for k in core:
        if k in summary:
            lines.append(f"{k}={float(summary[k]):.10f}")
    return "\n".join(lines)


def save_metrics_csv(metrics_list: List[Dict[str, float]], output_csv: Path) -> None:
    if not metrics_list:
        return

    fieldnames = [
        "index",
        "WR", "LWR", "DWS", "DWS_max", "num_cc",
        "L", "R", "T", "B",
        "cx", "cy",
        "D_conn", "frag_penalty",
        "S_frame", "S_side", "S_tb", "S_corner", "S_style", "S_max", "S_pos",
        "overlap_ratio", "Q_overlap", "beta_overlap",
        "dominant_style", "target_style",
        "use_coarse_ws", "dilation_px", "block_size", "block_occ_ratio", "connectivity",
        "metric_h", "metric_w",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, m in enumerate(metrics_list):
            row = {"index": i}
            for k in fieldnames[1:]:
                row[k] = m.get(k, 0.0)
            writer.writerow(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute whitespace/style metrics for generated_layouts.pkl"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to generated_layouts.pkl")
    parser.add_argument("--output_csv", type=str, required=True, help="Per-layout metric CSV path")
    parser.add_argument("--summary_json", type=str, required=True, help="Summary JSON path")
    parser.add_argument("--summary_txt", type=str, required=True, help="Summary TXT path")

    parser.add_argument("--box_format", type=str, default="cxcywh", choices=["cxcywh", "xywh", "xyxy"])
    parser.add_argument(
        "--ignore_labels",
        type=int,
        nargs="*",
        default=[5],
        help="Labels to exclude from whitespace metrics. Default ignores face label 5.",
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--beta_overlap",
        type=float,
        default=5.0,
        help="Q_overlap = exp(-beta_overlap * overlap_ratio)",
    )
    parser.add_argument(
        "--target_style",
        type=str,
        default="mean",
        choices=["mean", "max", "frame", "side", "tb", "corner", "right", "top", "hybrid"],
        help="Which style score to use as S_style in DWS. Aliases: right=side, top=tb, hybrid=corner.",
    )
    parser.add_argument("--large_thresh", type=float, default=0.95)
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])

    # Coarse whitespace options.
    parser.add_argument(
        "--use_coarse_ws",
        action="store_true",
        help="Enable content dilation + block downsampling before computing WR/LWR.",
    )
    parser.add_argument(
        "--dilation_px",
        type=int,
        default=2,
        help="Content dilation size. Start with 2 for 128x128 layouts.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4,
        help="Block size for coarse whitespace map. Start with 4 for 128x128 layouts.",
    )
    parser.add_argument(
        "--block_occ_ratio",
        type=float,
        default=0.15,
        help="Block is content if content ratio >= this value. Lower = stricter, fewer tiny gaps counted.",
    )

    parser.add_argument("--current_name", type=str, default="current")
    parser.add_argument("--compare_summary_json", type=str, default="")
    parser.add_argument("--compare_name", type=str, default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    summary_txt = Path(args.summary_txt)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.parent.mkdir(parents=True, exist_ok=True)

    layouts = load_generated_layouts(
        input_path,
        box_format=args.box_format,
        ignore_labels=list(args.ignore_labels),
    )

    metrics_list: List[Dict[str, float]] = []
    for boxes_xyxy, _labels_metric, boxes_all_xyxy, labels_all in layouts:
        metrics = whitespace_quality(
            boxes_xyxy,
            height=args.height,
            width=args.width,
            alpha=args.alpha,
            large_thresh=args.large_thresh,
            boxes_all_xyxy=boxes_all_xyxy,
            labels_all=labels_all,
            beta_overlap=args.beta_overlap,
            target_style=args.target_style,
            use_coarse_ws=args.use_coarse_ws,
            dilation_px=args.dilation_px,
            block_size=args.block_size,
            block_occ_ratio=args.block_occ_ratio,
            connectivity=args.connectivity,
        )
        metrics_list.append(metrics)

    save_metrics_csv(metrics_list, output_csv)
    curr_summary = summarize_metrics(metrics_list)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(curr_summary, f, indent=2, ensure_ascii=False)

    parts = [format_summary_block(args.current_name, curr_summary)]

    if args.compare_summary_json:
        with open(args.compare_summary_json, "r", encoding="utf-8") as f:
            base_summary = json.load(f)
        parts.append(format_summary_block(args.compare_name, base_summary))

        diff = diff_summary(curr_summary, base_summary)
        diff_lines = [f"=== {args.current_name} minus {args.compare_name} ==="]
        important_keys = [
            "mean_WR", "mean_LWR", "mean_D_conn", "mean_DWS", "mean_DWS_max",
            "mean_S_frame", "mean_S_side", "mean_S_tb", "mean_S_corner", "mean_S_style", "mean_S_max",
            "mean_overlap_ratio", "mean_Q_overlap",
            "mean_L", "mean_R", "mean_T", "mean_B",
            "ratio_frame", "ratio_side", "ratio_tb", "ratio_corner",
        ]
        for k in important_keys:
            if k in diff:
                diff_lines.append(f"{k}={diff[k]:.10f}")
        parts.append("\n".join(diff_lines))

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(parts) + "\n")

    print(f"[INFO] loaded layouts: {len(layouts)}")
    print(f"[INFO] saved csv: {output_csv}")
    print(f"[INFO] saved summary json: {summary_json}")
    print(f"[INFO] saved summary txt: {summary_txt}")
    print(f"[INFO] target_style: {normalize_target_style(args.target_style)}")
    print(f"[INFO] use_coarse_ws: {bool(args.use_coarse_ws)}")
    if args.use_coarse_ws:
        print(
            f"[INFO] coarse params: dilation_px={args.dilation_px}, "
            f"block_size={args.block_size}, block_occ_ratio={args.block_occ_ratio}"
        )


if __name__ == "__main__":
    main()
