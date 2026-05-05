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
- S_max   = max(S_frame, S_side, S_tb, S_corner)
- DWS_max = D_conn * frag_penalty * S_max

Keeps the original average-style score:
- S_style = (S_frame + S_side + S_tb + S_corner) / 4
- DWS     = D_conn * frag_penalty * S_style
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


def style_scores_from_margins(L: float, R: float, T: float, B: float) -> Dict[str, float]:
    margins = np.array([L, R, T, B], dtype=np.float32)
    mean_m = float(margins.mean())
    std_m = float(margins.std())

    # 1. 框架分數：維持原樣，評估整體四周留白的均勻度
    S_frame = mean_m - 0.5 * std_m

    horiz = np.array([L, R], dtype=np.float32)
    vert = np.array([T, B], dtype=np.float32)

    h_max, h_min, h_mean = float(horiz.max()), float(horiz.min()), float(horiz.mean())
    v_max, v_min, v_mean = float(vert.max()), float(vert.min()), float(vert.mean())

    # 1. 計算水平差異 (h_diff)
    h_diff = abs(L - R)
    
    # 2. 計算水平平衡得分 (S_h_balance)
    # 當 L 與 R 越接近（對稱），S_h_balance 越高
    S_h_balance = 1.0 / (1.0 + h_diff)
    
    # 3. 計算最小水平邊距 penalty (h_min_penalty)
    # 與 S_tb 邏輯一致，設定 0.2 為舒適邊距閾值
    h_min = min(L, R)
    # 如果最窄邊距小於 0.2，分數開始依比例衰減
    h_min_penalty = 1.0 if h_min >= 0.2 else (h_min / 0.2)

    # 4. 綜合評分 (重新分配權重)
    # 採用與 S_tb 類似的穩健權重：水平平均留白(50%) + 最大邊距(10%) + 對稱性(40%)
    # 降低 h_max 權重，避免獎勵 Baseline 那種單邊極大但另一邊貼邊的行為
    S_side_raw = (0.4 * h_mean + 0.1 * h_max + 0.5 * S_h_balance)
    
    # 5. 套用 Penalty
    S_side = S_side_raw * h_min_penalty

    # 3. 修改上下留白 (S_tb) - 這是你遇到的核心問題：
    # 改進後的模型 (improve.png) 分佈較散，v_min 會變大，導致 (v_max - v_min) 變小。
    # 1. 計算邊距平衡度 (你的現有邏輯)
    v_diff = abs(T - B)
    S_balance = 1.0 / (1.0 + v_diff)

    # 2. 新增：貼邊懲罰 (Penalty)
    # 如果最窄的邊距小於 0.2，則分數開始衰減
    v_min = min(T, B)
    penalty = 1.0 if v_min >= 0.2 else (v_min / 0.2)

    # 3. 重新分配權重：
    # 降低 v_max (避免獎勵極端偏移), 提升 S_balance, 並乘上 penalty
    # 建議權重：v_mean(40%) + v_max(10%) + S_balance(50%)
    S_tb_raw = (0.4 * v_mean + 0.1 * v_max + 0.5 * S_balance)
    S_tb = S_tb_raw * penalty

    # 數值修正與裁切
    S_frame = float(np.clip(S_frame, 0.0, 1.0))
    S_side = float(np.clip(S_side, 0.0, 1.0))
    S_tb = float(np.clip(S_tb, 0.0, 1.0))
    
    # 轉角分數：維持幾何平均
    S_corner = float(np.sqrt(max(0.0, S_side * S_tb)))
    
    # 最終風格得分
    S_style = float((S_frame + S_side + S_tb + S_corner) / 4.0)

    style_map = {
        "frame": S_frame,
        "side": S_side,
        "tb": S_tb,
        "corner": S_corner,
    }
    dominant_style = max(style_map, key=style_map.get)
    S_max = float(max(style_map.values()))

    return dict(
        S_frame=S_frame,
        S_side=S_side,
        S_tb=S_tb,
        S_corner=S_corner,
        S_style=S_style,
        S_max=S_max,
        S_pos=S_max,
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

    m["DWS"] = float(m["D_conn"] * m["frag_penalty"] * style["S_style"])
    m["DWS_max"] = float(m["D_conn"] * m["frag_penalty"] * style["S_max"])

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
    ))
    return m


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


SUMMARY_KEYS = [
    "WR", "LWR", "DWS", "DWS_max", "num_cc",
    "L", "R", "T", "B",
    "cx", "cy",
    "D_conn", "frag_penalty",
    "S_frame", "S_side", "S_tb", "S_corner", "S_style", "S_max", "S_pos",
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

    print(f"[INFO] loaded layouts: {len(layouts_xyxy)}")
    print(f"[INFO] saved csv: {output_csv}")
    print(f"[INFO] saved summary json: {summary_json}")
    print(f"[INFO] saved summary txt: {summary_txt}")


if __name__ == "__main__":
    main()
