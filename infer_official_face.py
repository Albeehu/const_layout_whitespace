"""
python infer_official.py \
--resume_ckpt final_eval/ckpts/right_final.pth \
--out_dir final_eval/official_infer/right \
--style right \
--n 100 \
--k 32 \
--seed 123 \
--max_fg_area 0.33 \
--max_elems 5

"""
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


STYLE_LABEL_POOLS = {
    "right": [
        [2, 1],
        [2, 1, 1],
        [2, 1, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0, 0],
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


def box_xyxy(box):
    cx, cy, w, h = box
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def intersection_area(box_a, box_b) -> float:
    a = box_xyxy(box_a)
    b = box_xyxy(box_b)

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return float(iw * ih)


def clip_boxes_xywh(boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    boxes = boxes.copy()

    for i, lb in enumerate(labels):
        if lb == 0:
            min_w, min_h = 0.04, 0.04
            max_w, max_h = 0.55, 0.55
        elif lb == 1:
            min_w, min_h = 0.16, 0.025
            max_w, max_h = 0.70, 0.18
        elif lb == 2:
            min_w, min_h = 0.16, 0.16
            max_w, max_h = 0.75, 0.75
        elif lb == 5:
            min_w, min_h = 0.06, 0.08
            max_w, max_h = 0.42, 0.50
        else:
            raise ValueError(f"Unexpected label {lb}; only 0/1/2/5 are allowed.")

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

    ov = 0.0
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            ov += intersection_area(boxes[i], boxes[j])
    return float(ov)


def union_bbox(boxes: np.ndarray):
    if len(boxes) == 0:
        return 0.0, 0.0, 0.0, 0.0
    x1 = np.min(boxes[:, 0] - boxes[:, 2] / 2.0)
    y1 = np.min(boxes[:, 1] - boxes[:, 3] / 2.0)
    x2 = np.max(boxes[:, 0] + boxes[:, 2] / 2.0)
    y2 = np.max(boxes[:, 1] + boxes[:, 3] / 2.0)
    return float(x1), float(y1), float(x2), float(y2)


def shrink_to_area_cap(boxes: np.ndarray, labels: np.ndarray, max_fg_area: float) -> np.ndarray:
    boxes = boxes.copy()
    area = total_fg_area(boxes)
    if area <= max_fg_area or area <= 1e-8:
        return boxes

    scale = math.sqrt(max_fg_area / area)
    for i, lb in enumerate(labels):
        if lb != 5:
            boxes[i, 2] *= scale
            boxes[i, 3] *= scale
    return boxes


def style_position_penalty(boxes: np.ndarray, style: str) -> float:
    x1, y1, x2, y2 = union_bbox(boxes)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    penalty = 0.0

    if style == "right":
        penalty += max(0.0, 0.40 - x1) * 3.0
        penalty += max(0.0, 0.55 - cx) * 2.0
        penalty += max(0.0, (x2 - x1) - 0.52) * 2.0

    elif style == "top":
        penalty += max(0.0, y2 - 0.62) * 3.0
        penalty += max(0.0, cy - 0.36) * 2.0
        penalty += max(0.0, (y2 - y1) - 0.45) * 2.0

    elif style == "hybrid":
        penalty += max(0.0, (x2 - x1) - 0.58) * 1.5
        penalty += max(0.0, (y2 - y1) - 0.52) * 1.5

    return float(penalty)


def choose_primary_image_index(boxes: np.ndarray, labels: np.ndarray):
    image_idxs = np.where(labels == 2)[0]
    if len(image_idxs) == 0:
        return None
    image_areas = boxes[image_idxs, 2] * boxes[image_idxs, 3]
    return int(image_idxs[np.argmax(image_areas)])


def derive_face_box_inside_image(
    img_box: np.ndarray,
    style: str,
    rng: np.random.RandomState,
):
    icx, icy, iw, ih = img_box

    if style == "right":
        fw = iw * rng.uniform(0.30, 0.44)
        fh = ih * rng.uniform(0.40, 0.60)
        rel_x = rng.uniform(-0.18, -0.02)
        rel_y = rng.uniform(-0.16, 0.02)

    elif style == "top":
        fw = iw * rng.uniform(0.28, 0.42)
        fh = ih * rng.uniform(0.38, 0.56)
        rel_x = rng.uniform(-0.06, 0.06)
        rel_y = rng.uniform(-0.18, -0.02)

    elif style == "hybrid":
        fw = iw * rng.uniform(0.28, 0.42)
        fh = ih * rng.uniform(0.38, 0.56)
        rel_x = rng.uniform(-0.10, 0.10)
        rel_y = rng.uniform(-0.14, 0.02)

    else:
        fw = iw * rng.uniform(0.28, 0.42)
        fh = ih * rng.uniform(0.38, 0.56)
        rel_x = rng.uniform(-0.08, 0.08)
        rel_y = rng.uniform(-0.14, -0.02)

    fcx = icx + rel_x * iw
    fcy = icy + rel_y * ih
    face = np.array([fcx, fcy, fw, fh], dtype=np.float32)

    img_x1, img_y1, img_x2, img_y2 = box_xyxy(img_box)
    fx1, fy1, fx2, fy2 = box_xyxy(face)

    if fx1 < img_x1:
        fcx += (img_x1 - fx1)
    if fy1 < img_y1:
        fcy += (img_y1 - fy1)
    if fx2 > img_x2:
        fcx -= (fx2 - img_x2)
    if fy2 > img_y2:
        fcy -= (fy2 - img_y2)

    return np.array([fcx, fcy, fw, fh], dtype=np.float32)


def append_face_if_needed(
    boxes: np.ndarray,
    labels: np.ndarray,
    add_face: bool,
    style: str,
    seed_offset: int = 0,
):
    if not add_face:
        return boxes, labels, None

    img_idx = choose_primary_image_index(boxes, labels)
    if img_idx is None:
        return boxes, labels, None

    rng = np.random.RandomState(seed_offset)
    face_box = derive_face_box_inside_image(
        boxes[img_idx],
        style=style,
        rng=rng,
    )

    new_boxes = np.concatenate([boxes, face_box[None, :]], axis=0)
    new_labels = np.concatenate([labels, np.array([5], dtype=np.int64)], axis=0)
    return new_boxes, new_labels, img_idx


def face_inside_image_penalty(boxes: np.ndarray, labels: np.ndarray) -> float:
    face_idxs = np.where(labels == 5)[0]
    if len(face_idxs) == 0:
        return 0.0

    img_idx = choose_primary_image_index(boxes, labels)
    if img_idx is None:
        return 100.0

    img = box_xyxy(boxes[img_idx])
    pen = 0.0

    for fi in face_idxs:
        face = box_xyxy(boxes[fi])
        pen += max(0.0, img[0] - face[0])
        pen += max(0.0, img[1] - face[1])
        pen += max(0.0, face[2] - img[2])
        pen += max(0.0, face[3] - img[3])

    return float(pen * 50.0)


def face_conflict_penalty(boxes: np.ndarray, labels: np.ndarray) -> float:
    face_idxs = np.where(labels == 5)[0]
    if len(face_idxs) == 0:
        return 0.0

    pen = 0.0
    for fi in face_idxs:
        for j, lb in enumerate(labels):
            if j == fi:
                continue
            if lb in [0, 1]:
                pen += intersection_area(boxes[fi], boxes[j])

    return float(pen)


def postprocess_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
    max_fg_area: float,
) -> np.ndarray:
    boxes = clip_boxes_xywh(boxes, labels)
    boxes = shrink_to_area_cap(boxes, labels, max_fg_area=max_fg_area)
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
    face_in_pen = face_inside_image_penalty(boxes, labels)
    face_conf_pen = face_conflict_penalty(boxes, labels)

    score = (
        10.0 * ov
        + 10.0 * area_violation
        + 2.5 * pos_pen
        + 20.0 * face_conf_pen
        + 1.0 * face_in_pen
    )

    meta = {
        "score": float(score),
        "overlap": float(ov),
        "area": float(area),
        "pos_penalty": float(pos_pen),
        "face_inside_penalty": float(face_in_pen),
        "face_conflict_penalty": float(face_conf_pen),
    }
    return score, meta


def choose_labels(style: str, max_elems: int, add_face: bool):
    pool = STYLE_LABEL_POOLS[style]
    gen_cap = max_elems - 1 if add_face else max_elems
    if gen_cap <= 0:
        raise ValueError("max_elems must be >= 1, and >= 2 if add_face is enabled.")

    valid = [seq for seq in pool if len(seq) <= gen_cap and 2 in seq]
    if len(valid) == 0:
        raise ValueError(
            f"No valid label preset for style={style}, max_elems={max_elems}, add_face={add_face}"
        )
    return np.array(random.choice(valid), dtype=np.int64)


def validate_custom_labels(labels, max_elems, add_face):
    if len(labels) == 0:
        raise ValueError("Custom labels cannot be empty.")

    bad = [x for x in labels if x not in [0, 1, 2]]
    if len(bad) > 0:
        raise ValueError(f"Custom generation labels can only contain 0/1/2. Bad labels: {bad}")

    total_visible = len(labels) + (1 if add_face else 0)
    if total_visible > max_elems:
        raise ValueError(
            f"Visible element count would be {total_visible}, but max_elems={max_elems}."
        )

    if add_face and 2 not in labels:
        raise ValueError("When add_face=1, custom labels must include at least one image (label 2).")


@torch.no_grad()
def sample_best_of_k(
    netG,
    labels_gen: np.ndarray,
    latent_size: int,
    k: int,
    device,
    style: str,
    max_fg_area: float,
    add_face: bool,
    sample_index: int,
):
    label_t = torch.tensor(labels_gen, dtype=torch.long, device=device).unsqueeze(0)
    padding_mask = torch.zeros_like(label_t, dtype=torch.bool)

    best_boxes = None
    best_labels = None
    best_meta = None

    for t in range(k):
        z = torch.randn(1, len(labels_gen), latent_size, device=device)
        boxes = netG(z, label_t, padding_mask)[0].detach().cpu().numpy()

        boxes = postprocess_boxes(
            boxes,
            labels=labels_gen,
            max_fg_area=max_fg_area,
        )

        cand_boxes, cand_labels, _ = append_face_if_needed(
            boxes,
            labels_gen,
            add_face=add_face,
            style=style,
            seed_offset=sample_index * 1000 + t,
        )

        cand_boxes = postprocess_boxes(
            cand_boxes,
            labels=cand_labels,
            max_fg_area=max_fg_area,
        )

        score, meta = score_layout(
            cand_boxes,
            labels=cand_labels,
            style=style,
            max_fg_area=max_fg_area,
        )

        if best_boxes is None or score < best_meta["score"]:
            best_boxes = cand_boxes
            best_labels = cand_labels
            best_meta = meta

    return best_boxes, best_labels, best_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--style", type=str, choices=["right", "hybrid", "top"], required=True)

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--max_fg_area", type=float, default=0.33)
    parser.add_argument("--max_elems", type=int, default=5)
    parser.add_argument("--add_face", type=int, default=1)
    parser.add_argument("--labels", type=int, nargs="*", default=None,
                        help="e.g. --labels 2 1 1 0")

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
            validate_custom_labels(args.labels, args.max_elems, bool(args.add_face))
            labels_gen = np.array(args.labels, dtype=np.int64)
        else:
            labels_gen = choose_labels(args.style, args.max_elems, bool(args.add_face))

        boxes, labels, meta = sample_best_of_k(
            netG=netG,
            labels_gen=labels_gen,
            latent_size=train_args["latent_size"],
            k=args.k,
            device=device,
            style=args.style,
            max_fg_area=args.max_fg_area,
            add_face=bool(args.add_face),
            sample_index=i,
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
    print("add_face:", args.add_face)
    if args.labels is not None:
        print("fixed generation labels:", args.labels)
        print("visible count =", len(args.labels) + (1 if args.add_face else 0))
    else:
        print("label pool mode: enabled")


if __name__ == "__main__":
    main()
