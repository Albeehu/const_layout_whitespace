#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
whitespace_metric_v4_max.py

For generated_layouts.pkl produced by current official inference scripts:
    results = [
        (boxes, labels),   # boxes usually in cx, cy, w, h
        ...
    ]

Adds:
- label-aware overlap ratio
- Q_overlap = exp(-beta_overlap * overlap_ratio)
- DWS       = D_conn * S_style * Q_overlap
- DWS_max   = D_conn * S_max * Q_overlap
- --target_style controls S_style: mean/max/frame/side/tb/corner, plus aliases right=side, top=tb, hybrid=corner

Notes:
- frag_penalty is still reported for analysis, but it is no longer multiplied into DWS.
- Face-image overlap is allowed when labels are available; other overlaps are penalized.
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


def whitespace_metrics_from_mask(
    whitespace_mask: np.ndarray,
    alpha: float = 0.1,
) -> Dict[str, float]:
    h, w = whitespace_mask.shape
    total_pixels = float(h * w)

    wr = float(whitespace_mask.sum()) / total_pixels
    if wr < 1e-8:
        return dict(
            WR=wr, LWR=0.0, DWS=0.0, DWS_max=0.0, num_cc=0.0,
            cx=0.5, cy=0.5,
            D_conn=0.0, frag_penalty=0.0
        )

    labeled, num_cc = cc_label(whitespace_mask)
    if num_cc == 0:
        return dict(
            WR=wr, LWR=0.0, DWS=0.0, DWS_max=0.0, num_cc=0.0,
            cx=0.5, cy=0.5,
            D_conn=0.0, frag_penalty=0.0
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

    D_conn = lwr / (wr + 1e-8) if wr > 1e-8 else 0.0
    frag_penalty = 1.0 / (1.0 + alpha * max(0.0, float(num_cc) - 1.0))

    return dict(
        WR=wr,
        LWR=lwr,
        DWS=0.0,
        DWS_max=0.0,
        num_cc=float(num_cc),
        cx=cx,
        cy=cy,
        D_conn=float(D_conn),
        frag_penalty=float(frag_penalty),
    )


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

    # Clip to canvas before area/overlap computation.
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


def normalize_target_style(target_style: str) -> str:
    """Normalize CLI target_style names to metric style names.

    Accepted values:
    - mean: average of frame/side/tb/corner
    - max: max of frame/side/tb/corner
    - frame, side, tb, corner: use the specified metric directly
    - right -> side, top -> tb, hybrid -> corner
    """
    target_style = str(target_style).lower().strip()
    alias = {
        "right": "side",
        "top": "tb",
        "hybrid": "corner",
    }
    target_style = alias.get(target_style, target_style)
    allowed = {"mean", "max", "frame", "side", "tb", "corner"}
    if target_style not in allowed:
        raise ValueError(f"Unsupported target_style={target_style}. Use one of {sorted(allowed)} or aliases: right/top/hybrid.")
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

    # 1. 框架分數：改成「四周留白 + 中偏上/中偏下」
    # 原本：S_frame = mean_m - 0.5 * std_m
    # 問題：原本主要獎勵四邊均勻留白，容易讓中偏上 / 中偏下構圖被扣分。
    #
    # 新版：
    # S_margin   = 四周留白的基本框架感
    # S_vertical = 上下留白差異，用來納入中偏上 / 中偏下
    # P_extreme  = 避免主體太靠上或太靠下
    #
    # T > B：內容偏下；B > T：內容偏上
    S_margin = mean_m - 0.5 * std_m
    S_vertical = abs(T - B)

    # 由上下邊界估計內容中心點 cy_content
    # cy_content < 0.5 代表內容偏上；cy_content > 0.5 代表內容偏下
    content_top = T
    content_bottom = 1.0 - B
    cy_content = (content_top + content_bottom) / 2.0

    # 允許中偏上 / 中偏下，但超過 0.25 後視為太貼邊並扣分
    vertical_tolerance = 0.25
    P_extreme = max(0.0, abs(cy_content - 0.5) - vertical_tolerance)

    # 推薦權重：保留 60% 原本框架感，新增 40% 上下偏移構圖感，並扣除極端靠邊
    S_frame = (0.6 * S_margin) + (0.4 * S_vertical) - (0.3 * P_extreme)

    horiz = np.array([L, R], dtype=np.float32)
    vert = np.array([T, B], dtype=np.float32)

    h_max, h_min, h_mean = float(horiz.max()), float(horiz.min()), float(horiz.mean())
    v_max, v_min, v_mean = float(vert.max()), float(vert.min()), float(vert.mean())

    # S_side：側邊留白分數
    # 公式：S_side = h_max + 0.8 * (h_max - h_min) + 0.2 * v_mean
    # 意義：獎勵左右其中一側的大留白，也獎勵左右不對稱形成的側邊構圖；
    #      但加入輕度貼邊懲罰，避免另一側過度貼邊仍拿太高分。
    h_diff = h_max - h_min
    S_side_raw = h_max + 0.8 * h_diff + 0.2 * v_mean

    # 輕度水平貼邊懲罰
    # h_min >= 0.05 不扣分；h_min < 0.05 時最低只乘 0.7。
    h_edge_threshold = 0.05
    h_edge_penalty = (
        1.0
        if h_min >= h_edge_threshold
        else 0.7 + 0.3 * (h_min / h_edge_threshold)
    )
    S_side = S_side_raw * h_edge_penalty

    # S_tb：上下留白分數
    # 對應 S_side 的垂直版本：
    # S_tb = v_max + 0.8 * (v_max - v_min) + 0.2 * h_mean
    # 意義：獎勵上下其中一側的大留白，也獎勵中偏上 / 中偏下構圖；
    #      但加入輕度貼邊懲罰，避免 top 或 bottom 過度貼邊仍拿太高分。
    v_diff = v_max - v_min
    S_tb_raw = v_max + 0.8 * v_diff + 0.2 * h_mean

    # 輕度垂直貼邊懲罰
    # v_min >= 0.05 不扣分；v_min < 0.05 時最低只乘 0.7。
    v_edge_threshold = 0.05
    v_edge_penalty = (
        1.0
        if v_min >= v_edge_threshold
        else 0.7 + 0.3 * (v_min / v_edge_threshold)
    )
    S_tb = S_tb_raw * v_edge_penalty

    # 數值修正與裁切
    S_frame = float(np.clip(S_frame, 0.0, 1.0))
    S_side = float(np.clip(S_side, 0.0, 1.0))
    S_tb = float(np.clip(S_tb, 0.0, 1.0))
    
    # 轉角分數：維持幾何平均
    S_corner = float(np.sqrt(max(0.0, S_side * S_tb)))
    
    style_map = {
        "frame": S_frame,
        "side": S_side,
        "tb": S_tb,
        "corner": S_corner,
    }
    dominant_style = max(style_map, key=style_map.get)
    S_max = float(max(style_map.values()))

    # 最終風格得分：可由 --target_style 指定
    # mean   = 平均四種風格
    # max    = 取四種風格最高分
    # frame  = 只看 S_frame
    # side   = 只看 S_side，也可用 alias right
    # tb     = 只看 S_tb，也可用 alias top
    # corner = 只看 S_corner，也可用 alias hybrid
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
) -> Dict[str, float]:
    occ_mask = layout_to_mask(boxes_xyxy, height=height, width=width, large_thresh=large_thresh)
    whitespace_mask = 1 - occ_mask

    m = whitespace_metrics_from_mask(whitespace_mask, alpha=alpha)
    L, R, T, B = content_margins_from_boxes(boxes_xyxy)
    style = style_scores_from_margins(L, R, T, B, target_style=target_style)

    overlap_boxes = boxes_all_xyxy if boxes_all_xyxy is not None else boxes_xyxy
    overlap_ratio = compute_overlap_ratio_xyxy(overlap_boxes, labels=labels_all)
    Q_overlap = float(np.exp(-float(beta_overlap) * overlap_ratio))

    # New requested definition:
    #   DWS = D_conn * S_style * Q_overlap
    # frag_penalty is still reported, but is intentionally not used here.
    m["DWS"] = float(m["D_conn"] * style["S_style"] * Q_overlap)
    m["DWS_max"] = float(m["D_conn"] * style["S_max"] * Q_overlap)

    m.update(dict(
        L=L, R=R, T=T, B=B,
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
    ))
    return m

def convert_boxes_to_xyxy(boxes: np.ndarray, box_format: str) -> np.ndarray:
    if box_format == "cxcywh":
        return cxcywh_to_xyxy(boxes)
    if box_format == "xywh":
        return xywh_topleft_to_xyxy(boxes)
    if box_format == "xyxy":
        return boxes.astype(np.float32)
    raise ValueError(f"Unsupported box_format: {box_format}")


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

SUMMARY_KEYS = [
    "WR", "LWR", "DWS", "DWS_max", "num_cc",
    "L", "R", "T", "B",
    "cx", "cy",
    "D_conn", "frag_penalty",
    "S_frame", "S_side", "S_tb", "S_corner", "S_style", "S_max", "S_pos",
    "overlap_ratio", "Q_overlap",
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

    core = [
        "mean_WR", "mean_LWR", "mean_DWS", "mean_DWS_max",
        "mean_S_frame", "mean_S_side", "mean_S_tb", "mean_S_corner", "mean_S_style", "mean_S_max",
        "mean_overlap_ratio", "mean_Q_overlap",
        "mean_L", "mean_R", "mean_T", "mean_B",
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
        "dominant_style",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, m in enumerate(metrics_list):
            row = {"index": i}
            for k in fieldnames[1:]:
                row[k] = m.get(k, 0.0)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare whitespace/style metrics for generated_layouts.pkl")
    parser.add_argument("--input", type=str, required=True, help="Path to generated_layouts.pkl")
    parser.add_argument("--output_csv", type=str, required=True, help="Per-layout metric CSV path")
    parser.add_argument("--summary_json", type=str, required=True, help="Summary JSON path")
    parser.add_argument("--summary_txt", type=str, required=True, help="Summary TXT path")
    parser.add_argument("--box_format", type=str, default="cxcywh", choices=["cxcywh", "xywh", "xyxy"])
    parser.add_argument("--ignore_labels", type=int, nargs="*", default=[5], help="Labels to exclude from whitespace metrics")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta_overlap", type=float, default=5.0,
                        help="Q_overlap = exp(-beta_overlap * overlap_ratio)")
    parser.add_argument(
        "--target_style",
        type=str,
        default="mean",
        choices=["mean", "max", "frame", "side", "tb", "corner", "right", "top", "hybrid"],
        help=(
            "Which style score to use as S_style in DWS. "
            "Aliases: right=side, top=tb, hybrid=corner."
        ),
    )
    parser.add_argument("--large_thresh", type=float, default=0.95)
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
            "mean_WR", "mean_LWR", "mean_DWS", "mean_DWS_max",
            "mean_S_frame", "mean_S_side", "mean_S_tb", "mean_S_corner", "mean_S_style", "mean_S_max",
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


if __name__ == "__main__":
    main()
