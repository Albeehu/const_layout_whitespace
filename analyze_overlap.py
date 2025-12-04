#計算各個label的IoU，用來看layout是否合理，會輸出pkl
import argparse
import pickle
from pathlib import Path

import numpy as np


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    boxes: (N, 4) [cx, cy, w, h] in [0,1]
    return: (N, 4) [x1, y1, x2, y2]
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    cx, cy, w, h = boxes.T
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def iou_pairs_in_layout(boxes_cxcywh: np.ndarray) -> np.ndarray:
    """
    boxes_cxcywh: (N, 4) [cx, cy, w, h]
    return: list of (i, j, iou) with i < j
    """
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
    N = boxes_xyxy.shape[0]
    results = []

    for i in range(N):
        x1_i, y1_i, x2_i, y2_i = boxes_xyxy[i]
        w_i = max(0.0, x2_i - x1_i)
        h_i = max(0.0, y2_i - y1_i)
        area_i = w_i * h_i
        if area_i <= 0:
            continue

        for j in range(i + 1, N):
            x1_j, y1_j, x2_j, y2_j = boxes_xyxy[j]
            w_j = max(0.0, x2_j - x1_j)
            h_j = max(0.0, y2_j - y1_j)
            area_j = w_j * h_j
            if area_j <= 0:
                continue

            ix1 = max(x1_i, x1_j)
            iy1 = max(y1_i, y1_j)
            ix2 = min(x2_i, x2_j)
            iy2 = min(y2_i, y2_j)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = area_i + area_j - inter
            iou = 0.0 if union <= 0 else inter / union
            results.append((i, j, iou))

    if not results:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(results, dtype=np.float32)


def parse_label_names(s: str, num_classes: int):
    """
    將 "SvgElement,TextElement,..." 轉成 list；不足的補 class_i
    """
    if not s:
        return [f"class_{i}" for i in range(num_classes)]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < num_classes:
        parts.extend([f"class_{i}" for i in range(len(parts), num_classes)])
    return parts[:num_classes]


def main():
    parser = argparse.ArgumentParser(
        description="分析 layout 中各類別 pair 的 IoU 分佈"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="輸入的 pkl 檔（list of (boxes, labels)）"
    )
    parser.add_argument(
        "--label_names", type=str, default="",
        help="用逗號分隔的 label 名稱，如："
             "'SvgElement,TextElement,ImageElement,ColoredBackground,SvgMaskElement'"
    )
    parser.add_argument(
        "--high_iou_thresh", type=float, default=0.7,
        help="判定『高重疊』的 IoU 門檻，預設 0.7"
    )
    parser.add_argument(
        "--min_count", type=int, default=20,
        help="只列出 pair 數量 >= min_count 的類別組合"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    print(f"[info] Loading layouts from {in_path} ...")
    with in_path.open("rb") as f:
        layouts = pickle.load(f)

    if not isinstance(layouts, list):
        raise ValueError("pkl 內容應該是 list of (boxes, labels)")

    # 收集所有 label，用來決定 num_classes
    all_labels = []
    for boxes, labels in layouts:
        labels = np.asarray(labels, dtype=np.int64)
        all_labels.append(labels)
    all_labels = np.concatenate(all_labels, axis=0)
    num_classes = int(all_labels.max()) + 1
    label_names = parse_label_names(args.label_names, num_classes)

    print(f"[info] num_classes = {num_classes}")
    print("[info] label names:")
    for i, name in enumerate(label_names):
        print(f"  {i}: {name}")

    # pair_stats[(a,b)] = list of IoU
    pair_stats = {}
    num_layouts = len(layouts)

    for idx, (boxes, labels) in enumerate(layouts):
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if boxes.shape[0] <= 1:
            continue

        pairs = iou_pairs_in_layout(boxes)  # (M, 3) [i, j, iou]
        for (i, j, iou) in pairs:
            la = int(labels[int(i)])
            lb = int(labels[int(j)])
            key = tuple(sorted((la, lb)))  # (小的, 大的)
            pair_stats.setdefault(key, []).append(float(iou))

        if (idx + 1) % 500 == 0:
            print(f"[info] processed {idx + 1}/{num_layouts} layouts")

    print("\n========== IoU statistics per label pair ==========")
    print(f"(只列出樣本數 >= {args.min_count} 的 pair)")
    print(f"high IoU threshold = {args.high_iou_thresh}\n")

    header = (
        "pair\tcount\tmean_iou\tp50\tp90\tp95\tp99\t"
        f"frac_iou>{args.high_iou_thresh}"
    )
    print(header)

    for (la, lb), vals in sorted(pair_stats.items()):
        vals = np.asarray(vals, dtype=np.float32)
        if vals.shape[0] < args.min_count:
            continue
        mean = float(vals.mean())
        p50 = float(np.percentile(vals, 50))
        p90 = float(np.percentile(vals, 90))
        p95 = float(np.percentile(vals, 95))
        p99 = float(np.percentile(vals, 99))
        frac_high = float(np.mean(vals > args.high_iou_thresh))

        name_a = label_names[la]
        name_b = label_names[lb]
        pair_name = f"({la}:{name_a}, {lb}:{name_b})"

        print(
            f"{pair_name}\t{vals.shape[0]}\t"
            f"{mean:.3f}\t{p50:.3f}\t{p90:.3f}\t{p95:.3f}\t{p99:.3f}\t"
            f"{frac_high:.3f}"
        )

    print("\n[done] overlap analysis finished.")


if __name__ == "__main__":
    main()
