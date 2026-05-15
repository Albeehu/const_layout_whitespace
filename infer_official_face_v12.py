# svg 有分大的跟小的 可以依據下的參數設定 推論用
"""
model 使用方式如下
python infer_official_face_v12.py \
  --resume_ckpt final_eval/ckpts/right_final.pth \
  --out_dir final_eval/official_infer/right_v2 \
  --style right \
  --n 100 \
  --k 64 \
  --max_fg_area 0.33 \
  --max_elems 5 \
  --add_face 1 \
  --svg_small_prob 0.72
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import random
import argparse
import pickle
import csv
import json
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
    "frame": [
        [2, 1],
        [2, 1, 1],
        [2, 1, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0, 0],
    ],
}

SVG_ROLE_ICON = "icon"
SVG_ROLE_TEXTURE = "texture"


# =========================
# Geometry helpers
# =========================
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


def boxes_without_face(boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    keep = labels != 5
    return boxes[keep]


def frame_margin_stats(boxes: np.ndarray, labels: np.ndarray):
    fg = boxes_without_face(boxes, labels)
    x1, y1, x2, y2 = union_bbox(fg)
    left = x1
    right = 1.0 - x2
    top = y1
    bottom = 1.0 - y2
    union_w = x2 - x1
    union_h = y2 - y1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return left, right, top, bottom, union_w, union_h, cx, cy


def frame_outer_band_violation(boxes: np.ndarray, labels: np.ndarray, band: float) -> float:
    pen = 0.0
    for box, lb in zip(boxes, labels):
        if lb == 5:
            continue
        x1, y1, x2, y2 = box_xyxy(box)
        pen += max(0.0, band - x1)
        pen += max(0.0, band - y1)
        pen += max(0.0, x2 - (1.0 - band))
        pen += max(0.0, y2 - (1.0 - band))
    return float(pen)


def frame_layout_penalty(
    boxes: np.ndarray,
    labels: np.ndarray,
    frame_band_margin: float,
    frame_target_w: float,
    frame_target_h: float,
    frame_max_ar: float,
) -> tuple[float, dict]:
    left, right, top, bottom, union_w, union_h, cx, cy = frame_margin_stats(boxes, labels)
    band_pen = frame_outer_band_violation(boxes, labels, band=frame_band_margin)

    min_margin_pen = (
        max(0.0, frame_band_margin - left)
        + max(0.0, frame_band_margin - right)
        + max(0.0, frame_band_margin - top)
        + max(0.0, frame_band_margin - bottom)
    )
    balance_pen = abs(left - right) + abs(top - bottom)
    center_pen = abs(cx - 0.5) + abs(cy - 0.5)
    width_pen = max(0.0, union_w - frame_target_w) + 0.6 * max(0.0, 0.16 - union_w)
    height_pen = max(0.0, union_h - frame_target_h) + 0.4 * max(0.0, 0.18 - union_h)
    aspect = union_h / max(union_w, 1e-6)
    aspect_pen = max(0.0, aspect - frame_max_ar)

    total = (
        6.0 * band_pen
        + 4.0 * min_margin_pen
        + 3.0 * balance_pen
        + 2.0 * center_pen
        + 2.5 * width_pen
        + 2.5 * height_pen
        + 2.0 * aspect_pen
    )
    meta = {
        "frame_band_penalty": float(band_pen),
        "frame_min_margin_penalty": float(min_margin_pen),
        "frame_balance_penalty": float(balance_pen),
        "frame_center_penalty": float(center_pen),
        "frame_width_penalty": float(width_pen),
        "frame_height_penalty": float(height_pen),
        "frame_aspect_penalty": float(aspect_pen),
        "frame_left": float(left),
        "frame_right": float(right),
        "frame_top": float(top),
        "frame_bottom": float(bottom),
        "frame_union_w": float(union_w),
        "frame_union_h": float(union_h),
        "frame_union_cx": float(cx),
        "frame_union_cy": float(cy),
        "frame_aspect": float(aspect),
    }
    return float(total), meta


# =========================
# SVG role control
# =========================
def sample_svg_roles(
    labels: np.ndarray,
    rng: np.random.RandomState,
    svg_small_prob: float,
):
    """
    label=0 的 SVG 會被指派成：
    - icon: 小型裝飾 / icon
    - texture: 較大的向量紋理 / decorative texture

    若同時有多個 SVG，強制至少出現一個 texture，並盡量讓剩下的偏 icon，
    這樣就不會全部都長成大塊 SVG。
    """
    roles = {}
    svg_idxs = np.where(labels == 0)[0].tolist()
    if len(svg_idxs) == 0:
        return roles

    if len(svg_idxs) == 1:
        roles[svg_idxs[0]] = SVG_ROLE_ICON if rng.rand() < svg_small_prob else SVG_ROLE_TEXTURE
        return roles

    rng.shuffle(svg_idxs)
    roles[svg_idxs[0]] = SVG_ROLE_TEXTURE
    roles[svg_idxs[1]] = SVG_ROLE_ICON

    for idx in svg_idxs[2:]:
        roles[idx] = SVG_ROLE_ICON if rng.rand() < 0.75 else SVG_ROLE_TEXTURE

    return roles


def get_svg_size_range(role: str):
    if role == SVG_ROLE_ICON:
        return (0.05, 0.18), (0.05, 0.18)
    if role == SVG_ROLE_TEXTURE:
        return (0.18, 0.52), (0.12, 0.42)
    return (0.04, 0.55), (0.04, 0.55)


def choose_svg_anchor(style: str, role: str, rng: np.random.RandomState):
    if role == SVG_ROLE_ICON:
        if style == "right":
            anchors = [(0.14, 0.16), (0.14, 0.82), (0.86, 0.14)]
        elif style == "top":
            anchors = [(0.12, 0.84), (0.88, 0.84), (0.14, 0.18)]
        elif style == "frame":
            anchors = [(0.24, 0.24), (0.76, 0.24), (0.24, 0.76), (0.76, 0.76)]
        else:  # hybrid
            anchors = [(0.14, 0.16), (0.86, 0.16), (0.14, 0.84), (0.86, 0.84)]
    else:
        if style == "right":
            anchors = [(0.18, 0.22), (0.20, 0.76), (0.82, 0.16)]
        elif style == "top":
            anchors = [(0.20, 0.80), (0.80, 0.82), (0.16, 0.24)]
        elif style == "frame":
            anchors = [(0.50, 0.30), (0.50, 0.70), (0.30, 0.50), (0.70, 0.50)]
        else:  # hybrid
            anchors = [(0.18, 0.22), (0.82, 0.22), (0.18, 0.78), (0.82, 0.78)]
    return anchors[rng.randint(len(anchors))]


def apply_svg_role_priors(
    boxes: np.ndarray,
    labels: np.ndarray,
    svg_roles: dict,
    style: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    對 SVG 做兩件事：
    1. 尺寸壓成 icon / texture 兩個尺度區間
    2. icon 偏向四角/邊緣，texture 偏向邊緣但保留較大覆蓋感
    """
    boxes = boxes.copy()

    for i, lb in enumerate(labels):
        if lb != 0:
            continue

        role = svg_roles.get(i, SVG_ROLE_TEXTURE)
        (min_w, max_w), (min_h, max_h) = get_svg_size_range(role)

        raw_w = float(boxes[i, 2])
        raw_h = float(boxes[i, 3])
        aspect = raw_w / max(raw_h, 1e-6)

        if role == SVG_ROLE_ICON:
            target_w = rng.uniform(0.06, 0.16)
            target_h = target_w / max(aspect, 0.6)
            target_h = np.clip(target_h, min_h, max_h)
            target_w = np.clip(target_h * aspect, min_w, max_w)
            target_h = np.clip(target_w / max(aspect, 0.6), min_h, max_h)
        else:
            target_w = np.clip(max(raw_w, rng.uniform(0.22, 0.42)), min_w, max_w)
            target_h = np.clip(max(raw_h, rng.uniform(0.14, 0.30)), min_h, max_h)

        boxes[i, 2] = target_w
        boxes[i, 3] = target_h

        ax, ay = choose_svg_anchor(style=style, role=role, rng=rng)

        if role == SVG_ROLE_ICON:
            # icon 直接更靠近角落，避免長成大塊主視覺
            boxes[i, 0] = 0.30 * boxes[i, 0] + 0.70 * ax + rng.uniform(-0.03, 0.03)
            boxes[i, 1] = 0.30 * boxes[i, 1] + 0.70 * ay + rng.uniform(-0.03, 0.03)
        else:
            # texture 保留一些模型原本自由度，只做輕推
            boxes[i, 0] = 0.65 * boxes[i, 0] + 0.35 * ax + rng.uniform(-0.05, 0.05)
            boxes[i, 1] = 0.65 * boxes[i, 1] + 0.35 * ay + rng.uniform(-0.05, 0.05)

    return boxes


# =========================
# Box post-processing
# =========================
def clip_boxes_xywh(boxes: np.ndarray, labels: np.ndarray, svg_roles: dict | None = None) -> np.ndarray:
    boxes = boxes.copy()
    svg_roles = svg_roles or {}

    for i, lb in enumerate(labels):
        if lb == 0:
            role = svg_roles.get(i, SVG_ROLE_TEXTURE)
            (min_w, max_w), (min_h, max_h) = get_svg_size_range(role)
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


def clip_boxes_xywh_margin(
    boxes: np.ndarray,
    labels: np.ndarray,
    svg_roles: dict | None = None,
    safe_margin: float = 0.065,
) -> np.ndarray:
    """Clip boxes while keeping a visible margin from canvas edges.

    This directly targets the whitespace metric edge penalties: the metric starts
    discounting side/top-bottom scores when the minimum margin is below ~0.05.
    """
    boxes = clip_boxes_xywh(boxes, labels, svg_roles=svg_roles)
    boxes = boxes.copy()

    for i in range(len(boxes)):
        half_w = boxes[i, 2] / 2.0
        half_h = boxes[i, 3] / 2.0

        # Adaptive fallback: do not create an invalid clipping range for large boxes.
        mx = min(float(safe_margin), max(0.0, (1.0 - float(boxes[i, 2])) / 2.0 - 1e-4))
        my = min(float(safe_margin), max(0.0, (1.0 - float(boxes[i, 3])) / 2.0 - 1e-4))

        boxes[i, 0] = np.clip(boxes[i, 0], mx + half_w, 1.0 - mx - half_w)
        boxes[i, 1] = np.clip(boxes[i, 1], my + half_h, 1.0 - my - half_h)

    return boxes


def _expanded_overlap(box_a: np.ndarray, box_b: np.ndarray, gap: float) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = box_xyxy(box_a)
    bx1, by1, bx2, by2 = box_xyxy(box_b)
    ox = min(ax2 + gap, bx2 + gap) - max(ax1 - gap, bx1 - gap)
    oy = min(ay2 + gap, by2 + gap) - max(ay1 - gap, by1 - gap)
    return float(max(0.0, ox)), float(max(0.0, oy))


def repel_overlaps(
    boxes: np.ndarray,
    labels: np.ndarray,
    gap_text: float = 0.035,
    gap_other: float = 0.025,
    steps: int = 30,
    safe_margin: float = 0.065,
    svg_roles: dict | None = None,
) -> np.ndarray:
    """Small deterministic physics pass to prevent overlaps and preserve gaps.

    - text/text gets a larger gap so words do not collide visually.
    - face/image is allowed, because face must live inside image.
    - face vs text/svg is not allowed and receives stronger separation.
    """
    boxes = boxes.copy()

    for _ in range(int(steps)):
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                li, lj = int(labels[i]), int(labels[j])

                if {li, lj} == {2, 5}:
                    continue

                if li == 5 or lj == 5:
                    gap = max(gap_text, 0.040)
                elif li == 1 and lj == 1:
                    gap = gap_text
                else:
                    gap = gap_other

                ox, oy = _expanded_overlap(boxes[i], boxes[j], gap=gap)
                if ox <= 0.0 or oy <= 0.0:
                    continue

                dx = float(boxes[i, 0] - boxes[j, 0])
                dy = float(boxes[i, 1] - boxes[j, 1])

                # Keep face fixed when possible; push the occluder instead.
                if li == 5 and lj != 2:
                    movable_i, movable_j = False, True
                elif lj == 5 and li != 2:
                    movable_i, movable_j = True, False
                else:
                    movable_i, movable_j = True, True

                if abs(dx) >= abs(dy):
                    push = ox / (2.0 if movable_i and movable_j else 1.0) + 1e-3
                    sgn = 1.0 if dx >= 0.0 else -1.0
                    if movable_i:
                        boxes[i, 0] += sgn * push
                    if movable_j:
                        boxes[j, 0] -= sgn * push
                else:
                    push = oy / (2.0 if movable_i and movable_j else 1.0) + 1e-3
                    sgn = 1.0 if dy >= 0.0 else -1.0
                    if movable_i:
                        boxes[i, 1] += sgn * push
                    if movable_j:
                        boxes[j, 1] -= sgn * push

                moved = True

        boxes = clip_boxes_xywh_margin(boxes, labels, svg_roles=svg_roles, safe_margin=safe_margin)
        if not moved:
            break

    return boxes


def keep_face_inside_primary_image(boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    boxes = boxes.copy()
    img_idx = choose_primary_image_index(boxes, labels)
    if img_idx is None:
        return boxes

    img_x1, img_y1, img_x2, img_y2 = box_xyxy(boxes[img_idx])
    face_idxs = np.where(labels == 5)[0].tolist()
    for fi in face_idxs:
        # Shrink face if the image became smaller after area cap.
        max_w = max(0.02, img_x2 - img_x1)
        max_h = max(0.02, img_y2 - img_y1)
        boxes[fi, 2] = min(float(boxes[fi, 2]), max_w * 0.92)
        boxes[fi, 3] = min(float(boxes[fi, 3]), max_h * 0.92)

        fx1, fy1, fx2, fy2 = box_xyxy(boxes[fi])
        if fx1 < img_x1:
            boxes[fi, 0] += img_x1 - fx1
        if fy1 < img_y1:
            boxes[fi, 1] += img_y1 - fy1
        if fx2 > img_x2:
            boxes[fi, 0] -= fx2 - img_x2
        if fy2 > img_y2:
            boxes[fi, 1] -= fy2 - img_y2

    return boxes


def fix_face_conflicts(
    boxes: np.ndarray,
    labels: np.ndarray,
    svg_roles: dict | None = None,
    safe_margin: float = 0.065,
    steps: int = 25,
) -> np.ndarray:
    """Ensure face overlaps only with image, never with text/SVG."""
    boxes = keep_face_inside_primary_image(boxes, labels)
    boxes = boxes.copy()
    face_idxs = np.where(labels == 5)[0].tolist()

    for _ in range(int(steps)):
        changed = False
        for fi in face_idxs:
            for j, lb in enumerate(labels):
                if j == fi or int(lb) == 2:
                    continue
                inter = intersection_area(boxes[fi], boxes[j])
                if inter <= 0.0:
                    continue

                dx = float(boxes[j, 0] - boxes[fi, 0])
                dy = float(boxes[j, 1] - boxes[fi, 1])
                if abs(dx) >= abs(dy):
                    shift = (boxes[fi, 2] + boxes[j, 2]) / 2.0 - abs(dx) + 0.040
                    boxes[j, 0] += (1.0 if dx >= 0.0 else -1.0) * shift
                else:
                    shift = (boxes[fi, 3] + boxes[j, 3]) / 2.0 - abs(dy) + 0.040
                    boxes[j, 1] += (1.0 if dy >= 0.0 else -1.0) * shift
                changed = True

        boxes = clip_boxes_xywh_margin(boxes, labels, svg_roles=svg_roles, safe_margin=safe_margin)
        boxes = keep_face_inside_primary_image(boxes, labels)
        if not changed:
            break

    return boxes


def shrink_text_to_avoid_collisions(
    boxes: np.ndarray,
    labels: np.ndarray,
    min_text_w: float = 0.13,
    max_iter: int = 8,
) -> np.ndarray:
    """Last-resort text shrink when text cannot be separated by movement alone."""
    boxes = boxes.copy()
    text_idxs = np.where(labels == 1)[0].tolist()
    if len(text_idxs) <= 1:
        return boxes

    for _ in range(int(max_iter)):
        changed = False
        for a_i, i in enumerate(text_idxs):
            for j in text_idxs[a_i + 1:]:
                if intersection_area(boxes[i], boxes[j]) <= 0.0:
                    continue
                if boxes[i, 2] > min_text_w:
                    boxes[i, 2] *= 0.94
                    changed = True
                if boxes[j, 2] > min_text_w:
                    boxes[j, 2] *= 0.94
                    changed = True
        if not changed:
            break
    return boxes


def shrink_to_area_cap(
    boxes: np.ndarray,
    labels: np.ndarray,
    max_fg_area: float,
) -> np.ndarray:
    """Shrink only non-face foreground to the requested area cap.

    Face is intentionally nested inside image and is ignored by whitespace metrics,
    so counting it in the area cap can shrink the image after face placement and
    create face/image violations.
    """
    boxes = boxes.copy()
    fg = boxes[labels != 5]
    area = total_fg_area(fg)
    if area <= max_fg_area or area <= 1e-8:
        return boxes

    scale = math.sqrt(max_fg_area / area)
    for i, lb in enumerate(labels):
        if lb != 5:
            boxes[i, 2] *= scale
            boxes[i, 3] *= scale
    return boxes


def postprocess_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
    max_fg_area: float,
    svg_roles: dict | None = None,
    style: str | None = None,
    rng: np.random.RandomState | None = None,
    use_svg_priors: bool = True,
    use_area_cap: bool = True,
    use_layout_repair: bool = True,
    safe_margin: float = 0.065,
    gap_text: float = 0.035,
    gap_other: float = 0.025,
) -> np.ndarray:
    clip_fn = clip_boxes_xywh_margin if use_layout_repair else clip_boxes_xywh

    if use_layout_repair:
        boxes = clip_fn(boxes, labels, svg_roles=svg_roles, safe_margin=safe_margin)
    else:
        boxes = clip_fn(boxes, labels, svg_roles=svg_roles)

    if use_svg_priors and svg_roles and style is not None and rng is not None:
        boxes = apply_svg_role_priors(
            boxes,
            labels=labels,
            svg_roles=svg_roles,
            style=style,
            rng=rng,
        )

    if use_layout_repair:
        boxes = clip_boxes_xywh_margin(boxes, labels, svg_roles=svg_roles, safe_margin=safe_margin)
    else:
        boxes = clip_boxes_xywh(boxes, labels, svg_roles=svg_roles)

    if use_area_cap:
        boxes = shrink_to_area_cap(boxes, labels, max_fg_area=max_fg_area)

    if use_layout_repair:
        boxes = clip_boxes_xywh_margin(boxes, labels, svg_roles=svg_roles, safe_margin=safe_margin)
        boxes = shrink_text_to_avoid_collisions(boxes, labels)
        boxes = repel_overlaps(
            boxes,
            labels,
            gap_text=gap_text,
            gap_other=gap_other,
            steps=30,
            safe_margin=safe_margin,
            svg_roles=svg_roles,
        )
    else:
        boxes = clip_boxes_xywh(boxes, labels, svg_roles=svg_roles)

    return boxes


# =========================
# Face logic
# =========================
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
    elif style == "frame":
        fw = iw * rng.uniform(0.28, 0.40)
        fh = ih * rng.uniform(0.38, 0.54)
        rel_x = rng.uniform(-0.05, 0.05)
        rel_y = rng.uniform(-0.10, 0.02)
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


# =========================
# Scoring
# =========================
def style_position_penalty(boxes: np.ndarray, style: str) -> float:
    """Style prior tuned for mean_DWS.

    Earlier versions pushed right/top too aggressively to maximize S_max-like
    layouts. For mean_DWS we want a compact group that is only moderately
    right/top biased while preserving all four margins.
    """
    x1, y1, x2, y2 = union_bbox(boxes)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    union_w = x2 - x1
    union_h = y2 - y1

    penalty = 0.0

    if style == "right":
        # Right means middle-right, not edge-right. Keep right-side whitespace.
        penalty += abs(cx - 0.64) * 1.8
        penalty += abs(cy - 0.50) * 1.2
        penalty += max(0.0, union_w - 0.46) * 3.0
        penalty += max(0.0, union_h - 0.62) * 2.0

        # Hard anti-edge prior; mean_DWS loses when any margin is too tiny.
        penalty += max(0.0, 0.07 - x1) * 4.0
        penalty += max(0.0, 0.07 - y1) * 4.0
        penalty += max(0.0, 0.07 - (1.0 - x2)) * 4.0
        penalty += max(0.0, 0.07 - (1.0 - y2)) * 4.0

    elif style == "top":
        # Top means upper-middle, not pasted to the top edge.
        penalty += abs(cx - 0.50) * 1.2
        penalty += abs(cy - 0.38) * 1.8
        penalty += max(0.0, union_w - 0.58) * 2.0
        penalty += max(0.0, union_h - 0.44) * 3.0

        penalty += max(0.0, 0.07 - x1) * 4.0
        penalty += max(0.0, 0.07 - y1) * 5.0
        penalty += max(0.0, 0.07 - (1.0 - x2)) * 4.0
        penalty += max(0.0, 0.07 - (1.0 - y2)) * 3.0

    elif style == "hybrid":
        penalty += max(0.0, union_w - 0.58) * 1.5
        penalty += max(0.0, union_h - 0.52) * 1.5

    elif style == "frame":
        penalty += abs(cx - 0.50) * 1.5
        penalty += abs(cy - 0.50) * 1.5
        penalty += max(0.0, union_w - 0.34) * 2.0
        penalty += max(0.0, union_h - 0.46) * 2.0

    return float(penalty)


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


def svg_role_penalty(
    boxes: np.ndarray,
    labels: np.ndarray,
    svg_roles: dict | None = None,
) -> float:
    svg_roles = svg_roles or {}
    svg_idxs = np.where(labels == 0)[0].tolist()
    if len(svg_idxs) == 0:
        return 0.0

    pen = 0.0
    icon_count = 0
    texture_count = 0

    for idx in svg_idxs:
        role = svg_roles.get(idx, SVG_ROLE_TEXTURE)
        cx, cy, w, h = boxes[idx]
        area = w * h
        edge_dist = min(cx, 1.0 - cx, cy, 1.0 - cy)

        if role == SVG_ROLE_ICON:
            icon_count += 1
            pen += max(0.0, w - 0.18) * 12.0
            pen += max(0.0, h - 0.18) * 12.0
            pen += max(0.0, edge_dist - 0.20) * 8.0
            pen += max(0.0, area - 0.030) * 30.0
        else:
            texture_count += 1
            pen += max(0.0, 0.16 - max(w, h)) * 10.0
            pen += max(0.0, 0.030 - area) * 40.0

    if len(svg_idxs) >= 2 and (icon_count == 0 or texture_count == 0):
        pen += 2.0

    return float(pen)


def margin_penalty(boxes: np.ndarray, labels: np.ndarray, safe_margin: float = 0.055) -> float:
    fg = boxes[labels != 5]
    if len(fg) == 0:
        return 0.0
    x1, y1, x2, y2 = union_bbox(fg)
    return float(
        max(0.0, safe_margin - x1)
        + max(0.0, safe_margin - y1)
        + max(0.0, safe_margin - (1.0 - x2))
        + max(0.0, safe_margin - (1.0 - y2))
    )


def alignment_penalty(boxes: np.ndarray, labels: np.ndarray) -> float:
    """Softly prefer neat stacks/columns without forcing every style to align."""
    fg_idxs = np.where(labels != 5)[0].tolist()
    if len(fg_idxs) <= 2:
        return 0.0

    centers_x = boxes[fg_idxs, 0]
    centers_y = boxes[fg_idxs, 1]
    lefts = boxes[fg_idxs, 0] - boxes[fg_idxs, 2] / 2.0
    rights = boxes[fg_idxs, 0] + boxes[fg_idxs, 2] / 2.0
    tops = boxes[fg_idxs, 1] - boxes[fg_idxs, 3] / 2.0
    bottoms = boxes[fg_idxs, 1] + boxes[fg_idxs, 3] / 2.0

    # Reward the best available axis/edge alignment; penalize only residual messiness.
    candidates = [
        float(np.std(centers_x)), float(np.std(centers_y)),
        float(np.std(lefts)), float(np.std(rights)),
        float(np.std(tops)), float(np.std(bottoms)),
    ]
    return float(min(candidates))


def whitespace_proxy_score(boxes: np.ndarray, labels: np.ndarray) -> float:
    """Fast proxy for mean_DWS during best-of-k rerank.

    The official whitespace_metric reports:
        DWS = D_conn * frag_penalty * S_style
    where S_style is the average of S_frame, S_side, S_tb, and S_corner.

    v11 optimized S_max-like behavior, which improved hybrid but hurt top/right
    mean_DWS. This v12 proxy optimizes the average-style score instead.
    """
    fg = boxes[labels != 5]
    if len(fg) == 0:
        return 0.0

    x1, y1, x2, y2 = union_bbox(fg)

    L = float(np.clip(x1, 0.0, 1.0))
    R = float(np.clip(1.0 - x2, 0.0, 1.0))
    T = float(np.clip(y1, 0.0, 1.0))
    B = float(np.clip(1.0 - y2, 0.0, 1.0))

    margins = np.array([L, R, T, B], dtype=np.float32)
    mean_m = float(margins.mean())
    std_m = float(margins.std())

    # Match whitespace_metric_v7 frame score structure.
    S_margin = mean_m - 0.5 * std_m
    S_vertical = abs(T - B)

    content_top = T
    content_bottom = 1.0 - B
    cy_content = 0.5 * (content_top + content_bottom)
    P_extreme = max(0.0, abs(cy_content - 0.5) - 0.25)

    S_frame = 0.6 * S_margin + 0.4 * S_vertical - 0.3 * P_extreme

    h_max = max(L, R)
    h_min = min(L, R)
    h_mean = 0.5 * (L + R)

    v_max = max(T, B)
    v_min = min(T, B)
    v_mean = 0.5 * (T + B)

    h_edge_thr = 0.055
    v_edge_thr = 0.055

    h_edge_penalty = 1.0 if h_min >= h_edge_thr else 0.7 + 0.3 * h_min / h_edge_thr
    v_edge_penalty = 1.0 if v_min >= v_edge_thr else 0.7 + 0.3 * v_min / v_edge_thr

    S_side = (h_max + 0.8 * (h_max - h_min) + 0.2 * v_mean) * h_edge_penalty
    S_tb = (v_max + 0.8 * (v_max - v_min) + 0.2 * h_mean) * v_edge_penalty
    S_corner = math.sqrt(max(0.0, S_side * S_tb))

    S_frame = float(np.clip(S_frame, 0.0, 1.0))
    S_side = float(np.clip(S_side, 0.0, 1.0))
    S_tb = float(np.clip(S_tb, 0.0, 1.0))
    S_corner = float(np.clip(S_corner, 0.0, 1.0))

    S_style = (S_frame + S_side + S_tb + S_corner) / 4.0

    # WR proxy: smaller foreground area generally leaves more whitespace.
    area = total_fg_area(fg)
    WR_proxy = float(np.clip(1.0 - area, 0.0, 1.0))

    union_w = max(1e-6, x2 - x1)
    union_h = max(1e-6, y2 - y1)

    # D_conn proxy: scattered foreground fragments split whitespace.
    scatter_pen = max(0.0, union_w - 0.58) + max(0.0, union_h - 0.62)
    D_conn_proxy = float(np.clip(1.0 - 0.8 * scatter_pen, 0.0, 1.0))

    # Fragment proxy: many visible boxes tend to create more connected components.
    n_fg = len(fg)
    frag_proxy = 1.0 / (1.0 + 0.06 * max(0, n_fg - 3))

    score = WR_proxy * D_conn_proxy * frag_proxy * S_style
    return float(np.clip(score, 0.0, 1.0))


def score_layout(
    boxes: np.ndarray,
    labels: np.ndarray,
    style: str,
    max_fg_area: float,
    svg_roles: dict | None = None,
    frame_band_margin: float = 0.10,
    frame_target_w: float = 0.28,
    frame_target_h: float = 0.38,
    frame_max_ar: float = 1.8,
    w_ov: float = 10.0,
    w_area: float = 10.0,
    w_pos: float = 2.5,
    w_face_conf: float = 20.0,
    w_face_in: float = 1.0,
    w_svg: float = 3.0,
    w_frame: float = 1.0,
    w_white: float = 0.0,
    w_margin: float = 0.0,
    w_align: float = 0.0,
):
    ov = pairwise_overlap_area(boxes)
    fg = boxes[labels != 5]
    area = total_fg_area(fg)
    area_violation = max(0.0, area - max_fg_area)
    pos_pen = style_position_penalty(fg if len(fg) else boxes, style)
    face_in_pen = face_inside_image_penalty(boxes, labels)
    face_conf_pen = face_conflict_penalty(boxes, labels)
    svg_pen = svg_role_penalty(boxes, labels, svg_roles=svg_roles)
    white_score = whitespace_proxy_score(boxes, labels)
    white_pen = 1.0 - white_score
    marg_pen = margin_penalty(boxes, labels, safe_margin=0.055)
    align_pen = alignment_penalty(boxes, labels)
    frame_pen = 0.0
    frame_meta = {}
    if style == "frame":
        frame_pen, frame_meta = frame_layout_penalty(
            boxes,
            labels,
            frame_band_margin=frame_band_margin,
            frame_target_w=frame_target_w,
            frame_target_h=frame_target_h,
            frame_max_ar=frame_max_ar,
        )

    score = (
        w_ov * ov
        + w_area * area_violation
        + w_pos * pos_pen
        + w_face_conf * face_conf_pen
        + w_face_in * face_in_pen
        + w_svg * svg_pen
        + w_frame * frame_pen
        + w_white * white_pen
        + w_margin * marg_pen
        + w_align * align_pen
    )

    meta = {
        "score": float(score),
        "overlap": float(ov),
        "area": float(area),
        "pos_penalty": float(pos_pen),
        "face_inside_penalty": float(face_in_pen),
        "face_conflict_penalty": float(face_conf_pen),
        "svg_penalty": float(svg_pen),
        "frame_penalty": float(frame_pen),
        "white_score": float(white_score),
        "white_penalty": float(white_pen),
        "margin_penalty": float(marg_pen),
        "alignment_penalty": float(align_pen),
        "svg_roles": {int(k): str(v) for k, v in (svg_roles or {}).items()},
        "w_ov": float(w_ov),
        "w_area": float(w_area),
        "w_pos": float(w_pos),
        "w_face_conf": float(w_face_conf),
        "w_face_in": float(w_face_in),
        "w_svg": float(w_svg),
        "w_frame": float(w_frame),
        "w_white": float(w_white),
        "w_margin": float(w_margin),
        "w_align": float(w_align),
    }
    meta.update(frame_meta)
    return score, meta


# =========================
# Label control
# =========================
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



def resolve_infer_mode(infer_mode: str, requested_k: int) -> dict:
    infer_mode = str(infer_mode).lower()

    if infer_mode not in ["raw", "final"]:
        raise ValueError(f"Unsupported infer_mode={infer_mode}. Use raw or final.")

    if infer_mode == "raw":
        return {
            "infer_mode": "raw",
            "effective_k": 1,
            "use_svg_priors": False,
            "use_area_cap": False,
            "use_layout_repair": False,
            "use_rerank": False,
            "w_ov": 0.0,
            "w_area": 0.0,
            "w_pos": 0.0,
            "w_face_conf": 0.0,
            "w_face_in": 0.0,
            "w_svg": 0.0,
            "w_frame": 0.0,
            "w_white": 0.0,
            "w_margin": 0.0,
            "w_align": 0.0,
        }

    return {
        "infer_mode": "final",
        "effective_k": int(requested_k),
        "use_svg_priors": True,
        "use_area_cap": True,
        "use_layout_repair": True,
        "use_rerank": True,

        # Hard constraints remain strong.
        "w_ov": 35.0,
        "w_area": 8.0,
        "w_face_conf": 100.0,
        "w_face_in": 8.0,

        # Lower style-position pressure; mean_DWS prefers balanced margins.
        "w_pos": 0.8,

        # Keep SVGs controlled without dominating whitespace rerank.
        "w_svg": 4.0,
        "w_frame": 1.0,

        # Main v12 change: mean_DWS-oriented rerank.
        "w_white": 24.0,
        "w_margin": 18.0,
        "w_align": 0.8,
    }


# =========================
# Best-of-k sampling
# =========================
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
    svg_small_prob: float,
    frame_band_margin: float,
    frame_target_w: float,
    frame_target_h: float,
    frame_max_ar: float,
    infer_mode: str = "final",
    safe_margin: float = 0.065,
    gap_text: float = 0.035,
    gap_other: float = 0.025,
):
    mode_cfg = resolve_infer_mode(infer_mode, requested_k=k)

    label_t = torch.tensor(labels_gen, dtype=torch.long, device=device).unsqueeze(0)
    padding_mask = torch.zeros_like(label_t, dtype=torch.bool)

    best_boxes = None
    best_labels = None
    best_meta = None

    for t in range(mode_cfg["effective_k"]):
        seed = sample_index * 1000 + t
        rng = np.random.RandomState(seed)

        z = torch.randn(1, len(labels_gen), latent_size, device=device)
        boxes = netG(z, label_t, padding_mask)[0].detach().cpu().numpy()

        svg_roles = sample_svg_roles(
            labels=labels_gen,
            rng=rng,
            svg_small_prob=svg_small_prob,
        )

        boxes = postprocess_boxes(
            boxes,
            labels=labels_gen,
            max_fg_area=max_fg_area,
            svg_roles=svg_roles,
            style=style,
            rng=rng,
            use_svg_priors=mode_cfg["use_svg_priors"],
            use_area_cap=mode_cfg["use_area_cap"],
            use_layout_repair=mode_cfg["use_layout_repair"],
            safe_margin=safe_margin,
            gap_text=gap_text,
            gap_other=gap_other,
        )

        cand_boxes, cand_labels, _ = append_face_if_needed(
            boxes,
            labels_gen,
            add_face=add_face,
            style=style,
            seed_offset=seed,
        )

        if mode_cfg["use_layout_repair"]:
            # Do not run area cap after appending face; it can shrink image and
            # leave the already-derived face outside image. Non-face area was
            # capped before appending face. Now repair face/text/svg constraints.
            cand_boxes = clip_boxes_xywh_margin(
                cand_boxes, cand_labels, svg_roles=svg_roles, safe_margin=safe_margin
            )
            cand_boxes = keep_face_inside_primary_image(cand_boxes, cand_labels)
            cand_boxes = fix_face_conflicts(
                cand_boxes, cand_labels, svg_roles=svg_roles, safe_margin=safe_margin, steps=25
            )
            cand_boxes = repel_overlaps(
                cand_boxes,
                cand_labels,
                gap_text=gap_text,
                gap_other=gap_other,
                steps=20,
                safe_margin=safe_margin,
                svg_roles=svg_roles,
            )
            cand_boxes = keep_face_inside_primary_image(cand_boxes, cand_labels)
        else:
            cand_boxes = clip_boxes_xywh(cand_boxes, cand_labels, svg_roles=svg_roles)

        score, meta = score_layout(
            cand_boxes,
            labels=cand_labels,
            style=style,
            max_fg_area=max_fg_area,
            svg_roles=svg_roles,
            frame_band_margin=frame_band_margin,
            frame_target_w=frame_target_w,
            frame_target_h=frame_target_h,
            frame_max_ar=frame_max_ar,
            w_ov=mode_cfg["w_ov"],
            w_area=mode_cfg["w_area"],
            w_pos=mode_cfg["w_pos"],
            w_face_conf=mode_cfg["w_face_conf"],
            w_face_in=mode_cfg["w_face_in"],
            w_svg=mode_cfg["w_svg"],
            w_frame=mode_cfg["w_frame"],
            w_white=mode_cfg["w_white"],
            w_margin=mode_cfg["w_margin"],
            w_align=mode_cfg["w_align"],
        )
        meta["infer_mode"] = mode_cfg["infer_mode"]
        meta["requested_k"] = int(k)
        meta["effective_k"] = int(mode_cfg["effective_k"])

        if not mode_cfg["use_rerank"]:
            return cand_boxes, cand_labels, meta

        if best_boxes is None or score < best_meta["score"]:
            best_boxes = cand_boxes
            best_labels = cand_labels
            best_meta = meta

    return best_boxes, best_labels, best_meta


def _safe_meta_value(meta: dict, key: str, default=0.0):
    val = meta.get(key, default)
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    return val


def compute_layout_face_cover_stats(boxes: np.ndarray, labels: np.ndarray) -> dict:
    """Compute layout/face coverage stats in the summary style the user wants.

    Occluders are TEXT(1) and SVG(0). IMAGE is not counted as occluder because
    face is intentionally inside image.
    """
    face_idxs = np.where(labels == 5)[0].tolist()
    occ_idxs = np.where(np.isin(labels, [0, 1]))[0].tolist()

    face_total_covs = []
    face_max_ious = []
    for fi in face_idxs:
        f_area = max(float(boxes[fi][2] * boxes[fi][3]), 1e-8)
        total_cov = 0.0
        max_iou = 0.0
        for oi in occ_idxs:
            inter = intersection_area(boxes[fi], boxes[oi])
            if inter <= 0.0:
                continue
            o_area = max(float(boxes[oi][2] * boxes[oi][3]), 1e-8)
            union = max(f_area + o_area - inter, 1e-8)
            iou = inter / union
            total_cov += inter / f_area
            if iou > max_iou:
                max_iou = iou
        face_total_covs.append(min(1.0, total_cov))
        face_max_ious.append(max_iou)

    if len(face_total_covs) == 0:
        return {
            'num_faces': 0,
            'layout_maxIoU': 0.0,
            'layout_meanCov': 0.0,
            'layout_maxCov': 0.0,
            'layout_anyCover': 0.0,
            'face_total_covs': [],
            'face_max_ious': [],
        }

    face_total_covs = np.asarray(face_total_covs, dtype=np.float64)
    face_max_ious = np.asarray(face_max_ious, dtype=np.float64)
    return {
        'num_faces': int(len(face_total_covs)),
        'layout_maxIoU': float(face_max_ious.max()),
        'layout_meanCov': float(face_total_covs.mean()),
        'layout_maxCov': float(face_total_covs.max()),
        'layout_anyCover': float((face_total_covs > 0.0).any()),
        'face_total_covs': face_total_covs.tolist(),
        'face_max_ious': face_max_ious.tolist(),
    }


def aggregate_cover_summary(results: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    layout_stats = [compute_layout_face_cover_stats(boxes, labels) for boxes, labels in results]
    num_layouts = len(layout_stats)
    num_faces = int(sum(x['num_faces'] for x in layout_stats))

    layout_maxIoUs = np.asarray([x['layout_maxIoU'] for x in layout_stats], dtype=np.float64) if num_layouts else np.zeros((0,), dtype=np.float64)
    layout_meanCovs = np.asarray([x['layout_meanCov'] for x in layout_stats], dtype=np.float64) if num_layouts else np.zeros((0,), dtype=np.float64)
    layout_maxCovs = np.asarray([x['layout_maxCov'] for x in layout_stats], dtype=np.float64) if num_layouts else np.zeros((0,), dtype=np.float64)
    layout_anyCover = np.asarray([x['layout_anyCover'] for x in layout_stats], dtype=np.float64) if num_layouts else np.zeros((0,), dtype=np.float64)

    all_face_covs = []
    for x in layout_stats:
        all_face_covs.extend(x['face_total_covs'])
    all_face_covs = np.asarray(all_face_covs, dtype=np.float64) if len(all_face_covs) else np.zeros((0,), dtype=np.float64)

    def mean_or_zero(arr):
        return float(arr.mean()) if arr.size > 0 else 0.0

    summary = {
        'num_layouts': int(num_layouts),
        'num_faces': int(num_faces),
        'layout_mean_maxIoU': mean_or_zero(layout_maxIoUs),
        'layout_mean_meanCov': mean_or_zero(layout_meanCovs),
        'layout_mean_maxCov': mean_or_zero(layout_maxCovs),
        'layout_anyCover_ratio': mean_or_zero(layout_anyCover),
        'face_cover_gt0_ratio': float((all_face_covs > 0.0).mean()) if all_face_covs.size > 0 else 0.0,
        'face_cover_gt005_ratio': float((all_face_covs > 0.05).mean()) if all_face_covs.size > 0 else 0.0,
        'face_cover_gt010_ratio': float((all_face_covs > 0.10).mean()) if all_face_covs.size > 0 else 0.0,
        'face_mean_coverage': mean_or_zero(all_face_covs),
    }
    return summary


def format_cover_summary_block(name: str, stats: dict) -> str:
    lines = [f'=== {name} ===']
    lines.append(f"num_layouts={int(stats.get('num_layouts', 0))} num_faces={int(stats.get('num_faces', 0))}")
    for key in [
        'layout_mean_maxIoU',
        'layout_mean_meanCov',
        'layout_mean_maxCov',
        'layout_anyCover_ratio',
        'face_cover_gt0_ratio',
        'face_cover_gt005_ratio',
        'face_cover_gt010_ratio',
        'face_mean_coverage',
    ]:
        lines.append(f"{key}={float(stats.get(key, 0.0)):.10f}")
    return '\n'.join(lines)


def diff_cover_summary(curr: dict, base: dict) -> dict:
    keys = [
        'layout_mean_maxIoU',
        'layout_mean_meanCov',
        'layout_mean_maxCov',
        'layout_anyCover_ratio',
        'face_cover_gt0_ratio',
        'face_cover_gt005_ratio',
        'face_cover_gt010_ratio',
        'face_mean_coverage',
    ]
    return {k: float(curr.get(k, 0.0)) - float(base.get(k, 0.0)) for k in keys}


def write_cover_summary_files(
    out_dir: Path,
    args,
    results: list[tuple[np.ndarray, np.ndarray]],
):
    current_name = args.summary_name if args.summary_name else out_dir.name
    current_stats = aggregate_cover_summary(results)

    stats_json_path = out_dir / 'summary_stats.json'
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(current_stats, f, indent=2, ensure_ascii=False)

    txt_path = out_dir / 'summary.txt'
    parts = [format_cover_summary_block(current_name, current_stats)]

    if args.compare_stats_json:
        with open(args.compare_stats_json, 'r', encoding='utf-8') as f:
            base_stats = json.load(f)
        base_name = args.compare_name if args.compare_name else Path(args.compare_stats_json).parent.name
        parts.append(format_cover_summary_block(base_name, base_stats))
        diff = diff_cover_summary(current_stats, base_stats)
        diff_lines = [f'=== {current_name} minus {base_name} ===']
        for k, v in diff.items():
            diff_lines.append(f'{k}: {v}')
        parts.append('\n'.join(diff_lines))

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(parts) + '\n')

    return current_stats


def write_metrics_csv(out_dir: Path, metas: list[dict], labels_list: list[np.ndarray], results: list[tuple[np.ndarray, np.ndarray]]):
    csv_path = out_dir / 'metrics.csv'
    fieldnames = [
        'index', 'filename', 'labels', 'visible_count',
        'score', 'overlap', 'area', 'pos_penalty',
        'face_inside_penalty', 'face_conflict_penalty',
        'svg_penalty', 'frame_penalty',
        'white_score', 'white_penalty', 'margin_penalty', 'alignment_penalty',
        'frame_band_penalty', 'frame_min_margin_penalty',
        'frame_balance_penalty', 'frame_center_penalty',
        'frame_width_penalty', 'frame_height_penalty',
        'frame_aspect_penalty',
        'frame_left', 'frame_right', 'frame_top', 'frame_bottom',
        'frame_union_w', 'frame_union_h',
        'frame_union_cx', 'frame_union_cy', 'frame_aspect',
        'layout_maxIoU', 'layout_meanCov', 'layout_maxCov', 'layout_anyCover',
        'num_faces', 'svg_roles',
    ]

    cover_stats = [compute_layout_face_cover_stats(boxes, labels) for boxes, labels in results]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (meta, labels, cstats) in enumerate(zip(metas, labels_list, cover_stats)):
            row = {
                'index': i,
                'filename': f'{i:04d}.png',
                'labels': ' '.join(map(str, labels.tolist())),
                'visible_count': int(len(labels)),
                'score': _safe_meta_value(meta, 'score'),
                'overlap': _safe_meta_value(meta, 'overlap'),
                'area': _safe_meta_value(meta, 'area'),
                'pos_penalty': _safe_meta_value(meta, 'pos_penalty'),
                'face_inside_penalty': _safe_meta_value(meta, 'face_inside_penalty'),
                'face_conflict_penalty': _safe_meta_value(meta, 'face_conflict_penalty'),
                'svg_penalty': _safe_meta_value(meta, 'svg_penalty'),
                'frame_penalty': _safe_meta_value(meta, 'frame_penalty'),
                'white_score': _safe_meta_value(meta, 'white_score'),
                'white_penalty': _safe_meta_value(meta, 'white_penalty'),
                'margin_penalty': _safe_meta_value(meta, 'margin_penalty'),
                'alignment_penalty': _safe_meta_value(meta, 'alignment_penalty'),
                'frame_band_penalty': _safe_meta_value(meta, 'frame_band_penalty'),
                'frame_min_margin_penalty': _safe_meta_value(meta, 'frame_min_margin_penalty'),
                'frame_balance_penalty': _safe_meta_value(meta, 'frame_balance_penalty'),
                'frame_center_penalty': _safe_meta_value(meta, 'frame_center_penalty'),
                'frame_width_penalty': _safe_meta_value(meta, 'frame_width_penalty'),
                'frame_height_penalty': _safe_meta_value(meta, 'frame_height_penalty'),
                'frame_aspect_penalty': _safe_meta_value(meta, 'frame_aspect_penalty'),
                'frame_left': _safe_meta_value(meta, 'frame_left'),
                'frame_right': _safe_meta_value(meta, 'frame_right'),
                'frame_top': _safe_meta_value(meta, 'frame_top'),
                'frame_bottom': _safe_meta_value(meta, 'frame_bottom'),
                'frame_union_w': _safe_meta_value(meta, 'frame_union_w'),
                'frame_union_h': _safe_meta_value(meta, 'frame_union_h'),
                'frame_union_cx': _safe_meta_value(meta, 'frame_union_cx'),
                'frame_union_cy': _safe_meta_value(meta, 'frame_union_cy'),
                'frame_aspect': _safe_meta_value(meta, 'frame_aspect'),
                'layout_maxIoU': float(cstats['layout_maxIoU']),
                'layout_meanCov': float(cstats['layout_meanCov']),
                'layout_maxCov': float(cstats['layout_maxCov']),
                'layout_anyCover': float(cstats['layout_anyCover']),
                'num_faces': int(cstats['num_faces']),
                'svg_roles': str(meta.get('svg_roles', {})),
            }
            writer.writerow(row)
# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--style", type=str, choices=["right", "hybrid", "top", "frame"], required=True)

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--max_fg_area", type=float, default=0.33)
    parser.add_argument("--max_elems", type=int, default=5)
    parser.add_argument("--add_face", type=int, default=1)
    parser.add_argument("--svg_small_prob", type=float, default=0.72,
                        help="single SVG 被視為小 icon 的機率")
    parser.add_argument("--infer_mode", type=str, choices=["raw", "final"], default="final",
                        help="raw=公平比較模型原生輸出；final=完整後處理與best-of-k最終系統結果")
    parser.add_argument("--labels", type=int, nargs="*", default=None,
                        help="e.g. --labels 2 1 1 0")
    parser.add_argument("--frame_band_margin", type=float, default=0.10)
    parser.add_argument("--frame_target_w", type=float, default=0.28)
    parser.add_argument("--frame_target_h", type=float, default=0.38)
    parser.add_argument("--frame_max_ar", type=float, default=1.8)
    parser.add_argument("--safe_margin", type=float, default=0.065,
                        help="final mode only: minimum canvas edge margin for all objects")
    parser.add_argument("--gap_text", type=float, default=0.035,
                        help="final mode only: target gap between text boxes")
    parser.add_argument("--gap_other", type=float, default=0.025,
                        help="final mode only: target gap between non-text boxes")
    parser.add_argument("--summary_name", type=str, default="")
    parser.add_argument("--compare_stats_json", type=str, default="")
    parser.add_argument("--compare_name", type=str, default="")

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
    labels_list = []

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
            svg_small_prob=args.svg_small_prob,
            frame_band_margin=args.frame_band_margin,
            frame_target_w=args.frame_target_w,
            frame_target_h=args.frame_target_h,
            frame_max_ar=args.frame_max_ar,
            infer_mode=args.infer_mode,
            safe_margin=args.safe_margin,
            gap_text=args.gap_text,
            gap_other=args.gap_other,
        )

        results.append((boxes, labels))
        metas.append(meta)
        labels_list.append(labels.copy())

        img = convert_layout_to_image(boxes, labels, dataset.colors, (120, 80))
        img.save(out_dir / f"{i:04d}.png")

    with open(out_dir / "generated_layouts.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(metas, f)

    write_metrics_csv(out_dir, metas, labels_list, results)
    current_stats = write_cover_summary_files(out_dir, args, results)

    print("Saved to:", out_dir)
    print("metrics csv:", out_dir / "metrics.csv")
    print("summary txt:", out_dir / "summary.txt")
    print("summary stats json:", out_dir / "summary_stats.json")
    print("style:", args.style)
    print("max_elems:", args.max_elems)
    print("max_fg_area:", args.max_fg_area)
    print("infer_mode:", args.infer_mode)
    print("requested best-of-k:", args.k)
    print("add_face:", args.add_face)
    print("svg_small_prob:", args.svg_small_prob)
    print("safe_margin:", args.safe_margin)
    print("gap_text:", args.gap_text)
    print("gap_other:", args.gap_other)
    if args.labels is not None:
        print("fixed generation labels:", args.labels)
        print("visible count =", len(args.labels) + (1 if args.add_face else 0))
    else:
        print("label pool mode: enabled")


if __name__ == "__main__":
    main()
