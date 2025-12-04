#用來挑選留白比例0.6的crello.pkl
import argparse
import pickle
from pathlib import Path

import numpy as np


def compute_whitespace_ratio(boxes: np.ndarray,
                             ignore_large: bool = True,
                             large_thresh: float = 0.95) -> float:
    """
    粗略計算留白比例：
        whitespace ≈ 1 - sum(w_i * h_i)

    boxes: (N, 4) 的 numpy array，[x, y, w, h]，皆為 0~1。
    ignore_large: 是否忽略「超大 box」（例如背景）。
    large_thresh: 面積 >= large_thresh 視為背景。
    """
    if boxes.size == 0:
        return 1.0  # 沒元素就當作全白

    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    if ignore_large:
        areas = areas[areas < large_thresh]

    covered = float(areas.sum())
    # clamp 到 [0, 1]
    covered = max(0.0, min(covered, 1.0))
    whitespace = 1.0 - covered
    whitespace = max(0.0, min(whitespace, 1.0))
    return whitespace


def main():
    parser = argparse.ArgumentParser(
        description="依留白比例篩選 layout，輸出成新的 pkl"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="輸入的 pkl 檔（list of (boxes, labels)）"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="輸出的 pkl 檔（篩選後）"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.6,
        help="留白比例門檻，例如 0.6"
    )
    parser.add_argument(
        "--mode", type=str, choices=["lt", "gt"], default="lt",
        help="lt: 保留 whitespace < threshold；gt: 保留 whitespace > threshold"
    )
    parser.add_argument(
        "--min_elements", type=int, default=1,
        help="保留的 layout 至少要有幾個元素"
    )
    parser.add_argument(
        "--ignore_large", action="store_true",
        help="忽略面積 >= large_thresh 的超大 box（視為背景）"
    )
    parser.add_argument(
        "--large_thresh", type=float, default=0.95,
        help="超大 box 面積門檻（normalized，預設 0.95）"
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"Loading layouts from {in_path} ...")
    with in_path.open("rb") as f:
        layouts = pickle.load(f)

    kept = []
    ws_all = []

    for boxes, labels in layouts:
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if boxes.shape[0] < args.min_elements:
            continue

        w_ratio = compute_whitespace_ratio(
            boxes,
            ignore_large=args.ignore_large,
            large_thresh=args.large_thresh,
        )
        ws_all.append(w_ratio)

        if args.mode == "lt":
            cond = w_ratio < args.threshold
        else:
            cond = w_ratio > args.threshold

        if cond:
            kept.append((boxes, labels))

    print(f"Total layouts: {len(layouts)}")
    print(f"Kept layouts : {len(kept)} "
          f"({len(kept) / max(len(layouts), 1):.2%})")
    if ws_all:
        ws_all = np.asarray(ws_all)
        print(f"Whitespace ratio: mean={ws_all.mean():.3f}, "
              f"min={ws_all.min():.3f}, max={ws_all.max():.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(kept, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved filtered layouts to {out_path}")


if __name__ == "__main__":
    main()
