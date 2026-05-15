# svg 有分大的跟小的 可以依據下的參數設定 推論用
"""
model 使用方式如下
python infer_official_face_v2.py \
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


def shrink_to_area_cap(
    boxes: np.ndarray,
    labels: np.ndarray,
    max_fg_area: float,
) -> np.ndarray:
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


def postprocess_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
    max_fg_area: float,
    svg_roles: dict | None = None,
    style: str | None = None,
    rng: np.random.RandomState | None = None,
    use_svg_priors: bool = True,
    use_area_cap: bool = True,
) -> np.ndarray:
    boxes = clip_boxes_xywh(boxes, labels, svg_roles=svg_roles)

    if use_svg_priors and svg_roles and style is not None and rng is not None:
        boxes = apply_svg_role_priors(
            boxes,
            labels=labels,
            svg_roles=svg_roles,
            style=style,
            rng=rng,
        )

    boxes = clip_boxes_xywh(boxes, labels, svg_roles=svg_roles)

    if use_area_cap:
        boxes = shrink_to_area_cap(boxes, labels, max_fg_area=max_fg_area)

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
    elif style == "frame":
        penalty += abs(cx - 0.50) * 1.5
        penalty += abs(cy - 0.50) * 1.5
        penalty += max(0.0, (x2 - x1) - 0.34) * 2.0
        penalty += max(0.0, (y2 - y1) - 0.46) * 2.0

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


def score_layout(
    boxes: np.ndarray,
    labels: np.ndarray,
    style: str,
    max_fg_area: float,
    svg_roles: dict | None = None,
    # 確保這裡有包含以下這些參數
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
):
    # 1. 基礎計算
    ov = pairwise_overlap_area(boxes)
    area = total_fg_area(boxes)
    area_violation = max(0.0, area - max_fg_area)
    x1, y1, x2, y2 = union_bbox(boxes)
    L, R, T, B = x1, 1.0 - x2, y1, 1.0 - y2
    
    # 2. 核心構圖邏輯 (針對 whitespace_metric 對齊)
    # 為了 DWS_max 高分，我們鼓勵內容「成群」並「靠邊」
    if style == "side":
        # 獎勵左右留白差異大，且不貼死邊
        pos_pen = (abs(L - R) * -2.0) + (max(0.0, 0.05 - min(L, R)) * 50.0)
    elif style == "tb":
        # 獎勵上下留白差異大
        pos_pen = (abs(T - B) * -2.0) + (max(0.0, 0.05 - min(T, B)) * 50.0)
    elif style == "hybrid":
        # 獎勵四角留白空間，將主體往中心偏置
        pos_pen = ((L + R + T + B) * -1.0)
    else: # frame
        # 獎勵均勻分佈
        pos_pen = (abs(L - R) + abs(T - B)) * 2.0

    # 3. 強制連通性獎勵 (針對 whitespace_metric 的 frag_penalty)
    # 我們懲罰過度分散的物件 (過多孤立區塊)
    # 檢查是否所有非 face 區塊是否過於分散 (簡易測量)
    dist_penalty = 0.0
    if len(boxes) > 2:
        # 簡單計算物件間的平均距離，太分散則懲罰
        centers = boxes[:, :2]
        dist_matrix = np.linalg.norm(centers[:, None] - centers, axis=-1)
        dist_penalty = np.mean(dist_matrix) * 2.0 

    # 4. 綜合得分
    score = (
        w_ov * ov * 5.0 + 
        w_area * area_violation * 10.0 + 
        w_pos * pos_pen + 
        w_svg * svg_role_penalty(boxes, labels, svg_roles) +
        dist_penalty 
    )
    
    meta = {"score": float(score), "pos_penalty": float(pos_pen), "dist_penalty": float(dist_penalty)}
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
            "use_rerank": False,
            "w_ov": 0.0,
            "w_area": 0.0,
            "w_pos": 0.0,
            "w_face_conf": 0.0,
            "w_face_in": 0.0,
            "w_svg": 0.0,
            "w_frame": 0.0,
        }

    return {
        "infer_mode": "final",
        "effective_k": int(requested_k),
        "use_svg_priors": True,
        "use_area_cap": True,
        "use_rerank": True,
        "w_ov": 10.0,
        "w_area": 10.0,
        "w_pos": 2.5,
        "w_face_conf": 20.0,
        "w_face_in": 1.0,
        "w_svg": 3.0,
        "w_frame": 1.0,
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
        )

        cand_boxes, cand_labels, _ = append_face_if_needed(
            boxes,
            labels_gen,
            add_face=add_face,
            style=style,
            seed_offset=seed,
        )

        cand_boxes = clip_boxes_xywh(cand_boxes, cand_labels, svg_roles=svg_roles)

        if mode_cfg["use_area_cap"]:
            cand_boxes = shrink_to_area_cap(
                cand_boxes,
                labels=cand_labels,
                max_fg_area=max_fg_area,
            )
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
    if args.labels is not None:
        print("fixed generation labels:", args.labels)
        print("visible count =", len(args.labels) + (1 if args.add_face else 0))
    else:
        print("label pool mode: enabled")


if __name__ == "__main__":
    main()
