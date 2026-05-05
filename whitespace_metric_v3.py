
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
whitespace_metric_generated_compare_allstyles.py

For generated_layouts.pkl produced by current official inference scripts:
    results = [
        (boxes, labels),   # boxes usually in cx, cy, w, h
        ...
    ]

This script:
1. loads generated layouts
2. optionally ignores labels (default suggested: face=5)
3. converts boxes to xyxy for mask computation
4. computes whitespace metrics per layout
5. summarizes ALL style scores:
   - S_frame
   - S_side
   - S_tb
   - S_corner
   - S_style
6. saves per-layout CSV + summary JSON/TXT
7. can compare current summary with a baseline summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import label as cc_label


# -----------------------------
# box helpers
# -----------------------------
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


# -----------------------------
# mask + metrics
# -----------------------------
def layout_to_mask(
    boxes_xyxy: np.ndarray,
    height: int = 128,
    width: int = 128,
    large_thresh: float = 0.95,
) -> np.ndarray:
    """
    Convert xyxy boxes to occupancy mask.
    1 = occupied, 0 = empty
    Boxes with normalized area >= large_thresh are ignored.
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
            WR=wr, LWR=0.0, DWS=0.0, num_cc=0.0,
            cx=0.5, cy=0.5,
            D_conn=0.0, frag_penalty=0.0
        )

    labeled, num_cc = cc_label(whitespace_mask)
    if num_cc == 0:
        return dict(
            WR=wr, LWR=0.0, DWS=0.0, num_cc=0.0,
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
        DWS=0.0,  # filled later after style score
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


def style_scores_from_margins(L: float, R: float, T: float, B: float) -> Dict[str, float]:
    margins = np.array([L, R, T, B], dtype=np.float32)
    mean_m = float(margins.mean())
    std_m = float(margins.std())

    # frame: all sides large and balanced
    S_frame = mean_m - 0.5 * std_m

    horiz = np.array([L, R], dtype=np.float32)
    vert = np.array([T, B], dtype=np.float32)

    h_max = float(horiz.max())
    h_min = float(horiz.min())
    v_max = float(vert.max())
    v_min = float(vert.min())
    v_mean = float(vert.mean())
    h_mean = float(horiz.mean())

    # one side especially large
    S_side = h_max + 0.8 * (h_max - h_min) + 0.2 * v_mean

    # top/bottom one side especially large
    S_tb = v_max + 0.8 * (v_max - v_min) + 0.2 * h_mean

    S_frame = float(np.clip(S_frame, 0.0, 1.0))
    S_side = float(np.clip(S_side, 0.0, 1.0))
    S_tb = float(np.clip(S_tb, 0.0, 1.0))

    # corner: both side and top/bottom strong at the same time
    S_corner = float(np.sqrt(max(0.0, S_side * S_tb)))

    # overall style summary
    S_style = float((S_frame + S_side + S_tb + S_corner) / 4.0)

    style_map = {
        "frame": S_frame,
        "side": S_side,
        "tb": S_tb,
        "corner": S_corner,
    }
    dominant_style = max(style_map, key=style_map.get)

    return dict(
        S_frame=S_frame,
        S_side=S_side,
        S_tb=S_tb,
        S_corner=S_corner,
        S_style=S_style,
        S_pos=max(style_map.values()),
        dominant_style=dominant_style,
    )


def whitespace_quality(
    boxes_xyxy: np.ndarray,
    height: int = 128,
    width: int = 128,
    alpha: float = 0.1,
    large_thresh: float = 0.95,
) -> Dict[str, float]:
    occ_mask = layout_to_mask(boxes_xyxy, height=height, width=width, large_thresh=large_thresh)
    whitespace_mask = 1 - occ_mask

    m = whitespace_metrics_from_mask(whitespace_mask, alpha=alpha)
    L, R, T, B = content_margins_from_boxes(boxes_xyxy)
    style = style_scores_from_margins(L, R, T, B)

    DWS = m["D_conn"] * m["frag_penalty"] * style["S_style"]
    m["DWS"] = float(DWS)

    m.update(dict(
        L=L, R=R, T=T, B=B,
        S_frame=style["S_frame"],
        S_side=style["S_side"],
        S_tb=style["S_tb"],
        S_corner=style["S_corner"],
        S_style=style["S_style"],
        S_pos=style["S_pos"],
        dominant_style=style["dominant_style"],
    ))
    return m


# -----------------------------
# loading
# -----------------------------
def load_generated_layouts(
    path: Path,
    box_format: str,
    ignore_labels: List[int],
) -> List[np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)

    layouts_xyxy: List[np.ndarray] = []

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

        boxes = filter_boxes_by_labels(boxes, labels, ignore_labels=ignore_labels)

        if box_format == "cxcywh":
            boxes_xyxy = cxcywh_to_xyxy(boxes)
        elif box_format == "xywh":
            boxes_xyxy = xywh_topleft_to_xyxy(boxes)
        elif box_format == "xyxy":
            boxes_xyxy = boxes.astype(np.float32)
        else:
            raise ValueError(f"Unsupported box_format: {box_format}")

        layouts_xyxy.append(boxes_xyxy)

    return layouts_xyxy


# -----------------------------
# summary
# -----------------------------
SUMMARY_KEYS = [
    "WR", "LWR", "DWS", "num_cc",
    "L", "R", "T", "B",
    "cx", "cy",
    "D_conn", "frag_penalty",
    "S_frame", "S_side", "S_tb", "S_corner", "S_style", "S_pos",
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
        "mean_WR", "mean_LWR", "mean_DWS",
        "mean_S_frame", "mean_S_side", "mean_S_tb", "mean_S_corner", "mean_S_style",
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
        "WR", "LWR", "DWS", "num_cc",
        "L", "R", "T", "B",
        "cx", "cy",
        "D_conn", "frag_penalty",
        "S_frame", "S_side", "S_tb", "S_corner", "S_style", "S_pos",
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


# -----------------------------
# CLI
# -----------------------------
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

    layouts_xyxy = load_generated_layouts(
        input_path,
        box_format=args.box_format,
        ignore_labels=list(args.ignore_labels),
    )

    metrics_list: List[Dict[str, float]] = []
    for boxes_xyxy in layouts_xyxy:
        metrics = whitespace_quality(
            boxes_xyxy,
            height=args.height,
            width=args.width,
            alpha=args.alpha,
            large_thresh=args.large_thresh,
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
            "mean_WR", "mean_LWR", "mean_DWS",
            "mean_S_frame", "mean_S_side", "mean_S_tb", "mean_S_corner", "mean_S_style",
            "mean_L", "mean_R", "mean_T", "mean_B",
            "ratio_frame", "ratio_side", "ratio_tb", "ratio_corner",
        ]
        for k in important_keys:
            if k in diff:
                diff_lines.append(f"{k}={diff[k]:.10f}")
        parts.append("\n".join(diff_lines))

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(parts) + "\n")

    print(f"[INFO] loaded layouts: {len(layouts_xyxy)}")
    print(f"[INFO] saved csv: {output_csv}")
    print(f"[INFO] saved summary json: {summary_json}")
    print(f"[INFO] saved summary txt: {summary_txt}")


if __name__ == "__main__":
    main()
