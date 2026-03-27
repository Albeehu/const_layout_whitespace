#純推論
import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import random
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from data import get_dataset
from util import convert_layout_to_image
from model.layoutganpp import Generator


# ------------------------------------------------------------
# seed
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# style-specific candidate label sequences
# only allow 0 / 1 / 2, and at most 5 elems
#
# 0 = svg/decor
# 1 = text
# 2 = image
# ------------------------------------------------------------
STYLE_LABEL_POOLS = {
    "right": [
        [2, 1],             # image + text
        [2, 1, 1],          # image + 2 text
        [2, 1, 0],          # image + text + decor
        [2, 1, 1, 0],       # image + 2 text + decor
        [2, 1, 1, 0, 0],    # image + 2 text + 2 decor
    ],
    "hybrid": [
        [2, 1],
        [2, 1, 1],
        [2, 1, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0, 0],
    ],
    "top": [
        [2, 1],
        [2, 1, 1],
        [2, 1, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0, 0],
    ],
}


# ------------------------------------------------------------
# geometry helpers
# boxes: (cx, cy, w, h), normalized to [0,1]
# ------------------------------------------------------------
def clip_boxes_xywh(boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    boxes = boxes.copy()

    for i, lb in enumerate(labels):
        if lb == 0:   # decor
            min_w, min_h = 0.04, 0.04
            max_w, max_h = 0.55, 0.55
        elif lb == 1: # text
            min_w, min_h = 0.16, 0.025
            max_w, max_h = 0.70, 0.18
        elif lb == 2: # image
            min_w, min_h = 0.16, 0.16
            max_w, max_h = 0.75, 0.75
        else:
            raise ValueError(f"Unexpected label {lb}; only 0/1/2 are allowed.")

        boxes[i, 2] = np.clip(boxes[i, 2], min_w, max_w)
        boxes[i, 3] = np.clip(boxes[i, 3], min_h, max_h)

        half_w = boxes[i, 2] / 2.0
        half_h = boxes[i, 3] / 2.0

        boxes[i, 0] = np.clip(boxes[i, 0], half_w, 1.0 - half_w)
        boxes[i, 1] = np.clip(boxes[i, 1], half_h, 1.0 - half_h)

    return boxes


def total_fg_area(boxes: np.ndarray) -> float:
    if len(boxes) == 0:
        return 0.0
    return float(np.sum(boxes[:, 2] * boxes[:, 3]))


def pairwise_overlap_area(boxes: np.ndarray) -> float:
    if len(boxes) <= 1:
        return 0.0

    x1 = boxes[:, 0] - boxes[:, 2] / 2.0
    y1 = boxes[:, 1] - boxes[:, 3] / 2.0
    x2 = boxes[:, 0] + boxes[:, 2] / 2.0
    y2 = boxes[:, 1] + boxes[:, 3] / 2.0

    ov = 0.0
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            ov += iw * ih
    return float(ov)


def union_bbox(boxes: np.ndarray):
    if len(boxes) == 0:
        return 0.0, 0.0, 0.0, 0.0
    x1 = np.min(boxes[:, 0] - boxes[:, 2] / 2.0)
    y1 = np.min(boxes[:, 1] - boxes[:, 3] / 2.0)
    x2 = np.max(boxes[:, 0] + boxes[:, 2] / 2.0)
    y2 = np.max(boxes[:, 1] + boxes[:, 3] / 2.0)
    return float(x1), float(y1), float(x2), float(y2)


def shrink_to_area_cap(boxes: np.ndarray, max_fg_area: float) -> np.ndarray:
    boxes = boxes.copy()
    area = total_fg_area(boxes)
    if area <= max_fg_area or area <= 1e-8:
        return boxes

    scale = math.sqrt(max_fg_area / area)
    boxes[:, 2] *= scale
    boxes[:, 3] *= scale
    return boxes


def style_position_penalty(boxes: np.ndarray, style: str) -> float:
    """
    encourage different content regions:
    - right: content should stay more on right side
    - top:   content should stay more on upper side
    - hybrid: lighter penalty, mostly compactness
    """
    x1, y1, x2, y2 = union_bbox(boxes)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    penalty = 0.0

    if style == "right":
        # penalize content drifting too far left or too wide
        penalty += max(0.0, 0.40 - x1) * 3.0
        penalty += max(0.0, 0.55 - cx) * 2.0
        penalty += max(0.0, (x2 - x1) - 0.52) * 2.0

    elif style == "top":
        # penalize content drifting too low or too tall
        penalty += max(0.0, y2 - 0.62) * 3.0
        penalty += max(0.0, cy - 0.36) * 2.0
        penalty += max(0.0, (y2 - y1) - 0.45) * 2.0

    elif style == "hybrid":
        # mild compactness
        penalty += max(0.0, (x2 - x1) - 0.58) * 1.5
        penalty += max(0.0, (y2 - y1) - 0.52) * 1.5

    return float(penalty)


def postprocess_boxes(boxes: np.ndarray, labels: np.ndarray, max_fg_area: float) -> np.ndarray:
    boxes = clip_boxes_xywh(boxes, labels)
    boxes = shrink_to_area_cap(boxes, max_fg_area=max_fg_area)
    boxes = clip_boxes_xywh(boxes, labels)
    return boxes


def score_layout(
    boxes: np.ndarray,
    labels: np.ndarray,
    style: str,
    max_fg_area: float
):
    ov = pairwise_overlap_area(boxes)
    area = total_fg_area(boxes)
    area_violation = max(0.0, area - max_fg_area)
    pos_pen = style_position_penalty(boxes, style)

    # encourage whitespace
    # smaller is better
    score = (
        12.0 * ov
        + 10.0 * area_violation
        + 2.5 * pos_pen
    )

    meta = {
        "score": float(score),
        "overlap": float(ov),
        "area": float(area),
        "pos_penalty": float(pos_pen),
    }
    return score, meta


def choose_labels(style: str, max_elems: int):
    pool = STYLE_LABEL_POOLS[style]
    valid = [seq for seq in pool if len(seq) <= max_elems]
    if len(valid) == 0:
        raise ValueError(f"No valid label preset for style={style}, max_elems={max_elems}")
    labels = random.choice(valid)
    return np.array(labels, dtype=np.int64)


@torch.no_grad()
def sample_best_of_k(
    netG,
    labels: np.ndarray,
    latent_size: int,
    k: int,
    device,
    style: str,
    max_fg_area: float,
):
    label_t = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
    padding_mask = torch.zeros_like(label_t, dtype=torch.bool)

    best_boxes = None
    best_meta = None

    for _ in range(k):
        z = torch.randn(1, len(labels), latent_size, device=device)
        bbox = netG(z, label_t, padding_mask)[0].detach().cpu().numpy()

        bbox = postprocess_boxes(
            bbox,
            labels=labels,
            max_fg_area=max_fg_area,
        )

        score, meta = score_layout(
            bbox,
            labels=labels,
            style=style,
            max_fg_area=max_fg_area,
        )

        if best_boxes is None or score < best_meta["score"]:
            best_boxes = bbox
            best_meta = meta

    return best_boxes, labels, best_meta


def validate_custom_labels(labels, max_elems):
    if len(labels) == 0:
        raise ValueError("Custom labels cannot be empty.")
    if len(labels) > max_elems:
        raise ValueError(f"Got {len(labels)} labels, but max_elems={max_elems}.")
    bad = [x for x in labels if x not in [0, 1, 2]]
    if len(bad) > 0:
        raise ValueError(f"Only labels 0/1/2 are allowed. Bad labels: {bad}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--style", type=str, choices=["right", "hybrid", "top"], required=True)

    parser.add_argument("--n", type=int, default=100, help="number of layouts to generate")
    parser.add_argument("--k", type=int, default=32, help="best-of-k")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--max_fg_area", type=float, default=0.33,
                        help="cap of total foreground area; smaller -> more whitespace")
    parser.add_argument("--max_elems", type=int, default=5,
                        help="maximum number of elements")
    parser.add_argument("--labels", type=int, nargs="*", default=None,
                        help="optional exact labels, e.g. --labels 2 1 1 0")

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.resume_ckpt, map_location=device)
    train_args = ckpt["args"]

    dataset = get_dataset(train_args["dataset"], "val", T.Compose([]))
    num_label = dataset.num_classes

    netG = Generator(
        train_args["latent_size"],
        num_label,
        d_model=train_args["G_d_model"],
        nhead=train_args["G_nhead"],
        num_layers=train_args["G_num_layers"],
    ).eval().requires_grad_(False).to(device)

    netG.load_state_dict(ckpt["netG"])

    results = []
    metas = []

    for i in tqdm(range(args.n), ncols=100):
        if args.labels is not None:
            validate_custom_labels(args.labels, args.max_elems)
            labels = np.array(args.labels, dtype=np.int64)
        else:
            labels = choose_labels(args.style, args.max_elems)

        boxes, labels, meta = sample_best_of_k(
            netG=netG,
            labels=labels,
            latent_size=train_args["latent_size"],
            k=args.k,
            device=device,
            style=args.style,
            max_fg_area=args.max_fg_area,
        )

        results.append((boxes, labels))
        metas.append(meta)

        img = convert_layout_to_image(boxes, labels, dataset.colors, (120, 80))
        img.save(out_dir / f"{i:04d}.png")

    with open(out_dir / "generated_layouts.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(metas, f)

    print("Saved to:", out_dir)
    print("style:", args.style)
    print("max_elems:", args.max_elems)
    print("max_fg_area:", args.max_fg_area)
    print("best-of-k:", args.k)
    if args.labels is not None:
        print("fixed labels:", args.labels)
    else:
        print("label pool mode: enabled")


if __name__ == "__main__":
    main()