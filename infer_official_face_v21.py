import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import random
import argparse
import pickle
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw
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
        [2, 2, 0],
        [2, 2, 0, 0],
    ],
    "hybrid": [
        [2, 1],
        [2, 1, 1],
        [2, 1, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0, 0],
        [2, 2, 0],
        [2, 2, 0, 0],
    ],
    "top": [
        [2, 1],
        [2, 1, 1],
        [2, 1, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0, 0],
        [2, 2, 0],
        [2, 2, 0, 0],
    ],
    "frame": [
        [2, 1],
        [2, 1, 1],
        [2, 1, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0, 0],
        [2, 2, 0],
        [2, 2, 0, 0],
    ],
}

SVG_ROLE_ICON = "icon"
SVG_ROLE_TEXTURE = "texture"


def box_xyxy(box):
    cx, cy, w, h = box
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def _as_rgb_color(c):
    try:
        arr = np.asarray(c).astype(float).flatten()
        if arr.size >= 3:
            if arr[:3].max() <= 1.0:
                arr = arr * 255.0
            return tuple(int(np.clip(v, 0, 255)) for v in arr[:3])
    except Exception:
        pass
    return (80, 120, 220)


def render_layout_preview_exact(
    boxes: np.ndarray,
    labels: np.ndarray,
    colors,
    preview_w: int,
    preview_h: int,
    border_width: int = 1,
    fill_alpha: int = 0,
) -> Image.Image:
    """Render layout preview with explicit width/height and thicker borders.

    This avoids the ambiguity of util.convert_layout_to_image size ordering.
    The output image is exactly preview_w x preview_h.
    """
    preview_w = int(preview_w)
    preview_h = int(preview_h)
    border_width = max(1, int(border_width))
    fill_alpha = int(np.clip(fill_alpha, 0, 255))

    canvas = Image.new("RGBA", (preview_w, preview_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for box, lb in zip(boxes, labels):
        lb = int(lb)
        try:
            base_color = _as_rgb_color(colors[lb])
        except Exception:
            base_color = (80, 120, 220)

        x1, y1, x2, y2 = box_xyxy(box)
        px1 = int(round(float(x1) * preview_w))
        py1 = int(round(float(y1) * preview_h))
        px2 = int(round(float(x2) * preview_w))
        py2 = int(round(float(y2) * preview_h))

        px1 = max(0, min(preview_w - 1, px1))
        py1 = max(0, min(preview_h - 1, py1))
        px2 = max(px1 + 1, min(preview_w, px2))
        py2 = max(py1 + 1, min(preview_h, py2))

        # Hollow / outline-only layout preview.
        # fill=None is intentional: no solid fill, only colored box outlines.
        draw.rectangle(
            [px1, py1, px2, py2],
            fill=None,
            outline=(*base_color, 255),
            width=border_width,
        )

    return canvas.convert("RGB")


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
):
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


def sample_svg_roles(labels: np.ndarray, rng: np.random.RandomState, svg_small_prob: float):
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


def get_svg_size_range(role: str, aspect: float = None, strict_asset_aspect: bool = False):
    """Return size bounds for SVG boxes.

    When strict_asset_aspect=True, we relax the minimum width/height so very thin
    or very tall SVG assets can preserve their true aspect ratio exactly. The
    role still determines only an upper bound on overall footprint.
    """
    if role == SVG_ROLE_ICON:
        base_min_w, base_max_w = 0.05, 0.18
        base_min_h, base_max_h = 0.05, 0.18
    elif role == SVG_ROLE_TEXTURE:
        base_min_w, base_max_w = 0.18, 0.52
        base_min_h, base_max_h = 0.12, 0.42
    else:
        base_min_w, base_max_w = 0.04, 0.55
        base_min_h, base_max_h = 0.04, 0.55

    if (not strict_asset_aspect) or aspect is None or aspect <= 0:
        return (base_min_w, base_max_w), (base_min_h, base_max_h)

    aspect = float(max(aspect, 1e-6))
    # Important: do not force incompatible minima for extreme aspect ratios.
    # We keep only soft/very small minima so exact aspect can be preserved.
    min_w = 0.008
    min_h = 0.008
    max_w = float(base_max_w)
    max_h = float(base_max_h)
    return (min_w, max_w), (min_h, max_h)


def choose_svg_anchor(style: str, role: str, rng: np.random.RandomState):
    if role == SVG_ROLE_ICON:
        if style == "right":
            anchors = [(0.14, 0.16), (0.14, 0.82), (0.86, 0.14)]
        elif style == "top":
            anchors = [(0.12, 0.84), (0.88, 0.84), (0.14, 0.18)]
        elif style == "frame":
            anchors = [(0.24, 0.24), (0.76, 0.24), (0.24, 0.76), (0.76, 0.76)]
        else:
            anchors = [(0.14, 0.16), (0.86, 0.16), (0.14, 0.84), (0.86, 0.84)]
    else:
        if style == "right":
            anchors = [(0.18, 0.22), (0.20, 0.76), (0.82, 0.16)]
        elif style == "top":
            anchors = [(0.20, 0.80), (0.80, 0.82), (0.16, 0.24)]
        elif style == "frame":
            anchors = [(0.50, 0.30), (0.50, 0.70), (0.30, 0.50), (0.70, 0.50)]
        else:
            anchors = [(0.18, 0.22), (0.82, 0.22), (0.18, 0.78), (0.82, 0.78)]
    return anchors[rng.randint(len(anchors))]


def apply_svg_role_priors(boxes, labels, svg_roles, style, rng, allow_size_change: bool = True):
    boxes = boxes.copy()
    for i, lb in enumerate(labels):
        if lb != 0:
            continue
        role = svg_roles.get(i, SVG_ROLE_TEXTURE)
        raw_w = float(boxes[i, 2])
        raw_h = float(boxes[i, 3])
        aspect = raw_w / max(raw_h, 1e-6)
        (min_w, max_w), (min_h, max_h) = get_svg_size_range(role)
        if allow_size_change:
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
            boxes[i, 0] = 0.30 * boxes[i, 0] + 0.70 * ax + rng.uniform(-0.03, 0.03)
            boxes[i, 1] = 0.30 * boxes[i, 1] + 0.70 * ay + rng.uniform(-0.03, 0.03)
        else:
            boxes[i, 0] = 0.65 * boxes[i, 0] + 0.35 * ax + rng.uniform(-0.05, 0.05)
            boxes[i, 1] = 0.65 * boxes[i, 1] + 0.35 * ay + rng.uniform(-0.05, 0.05)
    return boxes


def clip_boxes_xywh(boxes: np.ndarray, labels: np.ndarray, svg_roles=None) -> np.ndarray:
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


def clip_boxes_xywh_margin(boxes, labels, svg_roles=None, safe_margin: float = 0.065):
    boxes = clip_boxes_xywh(boxes, labels, svg_roles=svg_roles)
    boxes = boxes.copy()
    for i in range(len(boxes)):
        half_w = boxes[i, 2] / 2.0
        half_h = boxes[i, 3] / 2.0
        mx = min(float(safe_margin), max(0.0, (1.0 - float(boxes[i, 2])) / 2.0 - 1e-4))
        my = min(float(safe_margin), max(0.0, (1.0 - float(boxes[i, 3])) / 2.0 - 1e-4))
        boxes[i, 0] = np.clip(boxes[i, 0], mx + half_w, 1.0 - mx - half_w)
        boxes[i, 1] = np.clip(boxes[i, 1], my + half_h, 1.0 - my - half_h)
    return boxes


def _expanded_overlap(box_a, box_b, gap: float):
    ax1, ay1, ax2, ay2 = box_xyxy(box_a)
    bx1, by1, bx2, by2 = box_xyxy(box_b)
    ox = min(ax2 + gap, bx2 + gap) - max(ax1 - gap, bx1 - gap)
    oy = min(ay2 + gap, by2 + gap) - max(ay1 - gap, by1 - gap)
    return float(max(0.0, ox)), float(max(0.0, oy))


def repel_overlaps(boxes, labels, gap_text=0.035, gap_other=0.025, steps=30, safe_margin=0.065, svg_roles=None):
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
                if abs(dx) >= abs(dy):
                    push = ox / 2.0 + 1e-3
                    sgn = 1.0 if dx >= 0.0 else -1.0
                    boxes[i, 0] += sgn * push
                    boxes[j, 0] -= sgn * push
                else:
                    push = oy / 2.0 + 1e-3
                    sgn = 1.0 if dy >= 0.0 else -1.0
                    boxes[i, 1] += sgn * push
                    boxes[j, 1] -= sgn * push
                moved = True
        boxes = clip_boxes_xywh_margin(boxes, labels, svg_roles=svg_roles, safe_margin=safe_margin)
        if not moved:
            break
    return boxes


def shrink_text_to_avoid_collisions(boxes: np.ndarray, labels: np.ndarray, min_text_w: float = 0.13, max_iter: int = 8):
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


def shrink_to_area_cap(boxes: np.ndarray, labels: np.ndarray, max_fg_area: float) -> np.ndarray:
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
    boxes,
    labels,
    max_fg_area,
    svg_roles=None,
    style=None,
    rng=None,
    use_svg_priors=True,
    use_area_cap=True,
    use_layout_repair=True,
    safe_margin=0.065,
    gap_text=0.035,
    gap_other=0.025,
    svg_role_affects_size=True,
):
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
            allow_size_change=svg_role_affects_size,
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
        boxes = repel_overlaps(boxes, labels, gap_text=gap_text, gap_other=gap_other, steps=30, safe_margin=safe_margin, svg_roles=svg_roles)
    else:
        boxes = clip_boxes_xywh(boxes, labels, svg_roles=svg_roles)
    return boxes


def detect_faces_yolov8_norm(image_path: str, model_path: str = "yolov8n-face.pt", conf: float = 0.3, max_faces: int = 1):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics is required. Please run: pip install ultralytics") from e
    img = Image.open(str(image_path)).convert("RGB")
    W, H = img.size
    model = YOLO(str(model_path))
    results = model(str(image_path), conf=conf, verbose=False)
    faces = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)
            score = float(box.conf[0].detach().cpu().numpy())
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            faces.append({
                "xywh_norm": [float(cx), float(cy), float(bw / W), float(bh / H)],
                "conf": score,
                "area": float((bw / W) * (bh / H)),
            })
    faces = sorted(faces, key=lambda x: (x["conf"], x["area"]), reverse=True)
    if max_faces and max_faces > 0:
        faces = faces[:int(max_faces)]
    return [f["xywh_norm"] for f in faces]


def get_user_image_aspect(image_path: str):
    if not image_path:
        return None
    img = Image.open(str(image_path)).convert("RGB")
    W, H = img.size
    if H <= 0:
        return None
    return float(W) / float(H)

def _parse_svg_length(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    num = []
    for ch in s:
        if ch.isdigit() or ch in ['.', '-', '+', 'e', 'E']:
            num.append(ch)
        else:
            break
    try:
        return float(''.join(num)) if num else None
    except Exception:
        return None


def get_svg_aspect(svg_path: str):
    """Return aspect ratio W/H for SVG or raster decorative assets.

    SVG_PATHS can contain true .svg files or rasterized SVG assets such as .png.
    For SVG, width/height is preferred; viewBox is used as fallback.
    For PNG/JPG/WebP, Pillow is used to read the image size.
    """
    if not svg_path:
        return None

    path = Path(str(svg_path))
    suffix = path.suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        try:
            img = Image.open(str(path)).convert("RGBA")
            w, h = img.size
            if h > 0:
                return float(w) / float(h)
        except Exception:
            return None

    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(str(path)).getroot()
    except Exception:
        # Last fallback: try to read it as a normal image even if extension is unusual.
        try:
            img = Image.open(str(path)).convert("RGBA")
            w, h = img.size
            if h > 0:
                return float(w) / float(h)
        except Exception:
            pass
        return None

    width = _parse_svg_length(root.attrib.get('width'))
    height = _parse_svg_length(root.attrib.get('height'))
    if width and height and height > 0:
        return float(width) / float(height)

    view_box = root.attrib.get('viewBox') or root.attrib.get('viewbox')
    if view_box:
        parts = str(view_box).replace(',', ' ').split()
        if len(parts) == 4:
            try:
                vb_w = float(parts[2])
                vb_h = float(parts[3])
                if vb_h > 0:
                    return float(vb_w) / float(vb_h)
            except Exception:
                pass
    return None


def collect_svg_inputs(svg_paths):
    infos = []
    for p in svg_paths:
        p = str(p)
        infos.append({
            'path': p,
            'aspect': get_svg_aspect(p),
        })
    return infos


def collect_image_inputs(image_paths, face_model: str, face_conf: float, max_detected_faces: int):
    infos = []
    for p in image_paths:
        p = str(p)
        aspect = get_user_image_aspect(p)
        faces = detect_faces_yolov8_norm(
            image_path=p,
            model_path=face_model,
            conf=face_conf,
            max_faces=max_detected_faces,
        )
        infos.append({
            "path": p,
            "aspect": aspect,
            "faces": faces,
        })
    return infos


def enforce_image_aspect_ratios(
    boxes: np.ndarray,
    labels: np.ndarray,
    image_infos,
    preserve_image_aspect: bool = True,
    min_w: float = 0.16,
    min_h: float = 0.16,
    max_w: float = 0.75,
    max_h: float = 0.75,
    safe_margin: float = 0.065,
):
    if (not preserve_image_aspect) or not image_infos:
        return boxes
    boxes = boxes.copy()
    image_idxs = np.where(labels == 2)[0].tolist()
    if len(image_idxs) == 0:
        return boxes
    for local_i, idx in enumerate(image_idxs):
        aspect = image_infos[local_i % len(image_infos)].get("aspect", None)
        if aspect is None or aspect <= 0:
            continue
        aspect = float(np.clip(float(aspect), 0.25, 4.0))
        cx, cy, w, h = [float(v) for v in boxes[idx]]
        area = max(w * h, 1e-6)
        new_w = math.sqrt(area * aspect)
        new_h = math.sqrt(area / aspect)
        if new_w > max_w:
            new_w = max_w
            new_h = new_w / aspect
        if new_h > max_h:
            new_h = max_h
            new_w = new_h * aspect
        if new_w < min_w:
            new_w = min_w
            new_h = new_w / aspect
        if new_h < min_h:
            new_h = min_h
            new_w = new_h * aspect
        new_w = float(np.clip(new_w, min_w, max_w))
        new_h = float(np.clip(new_h, min_h, max_h))
        half_w = new_w / 2.0
        half_h = new_h / 2.0
        mx = min(float(safe_margin), max(0.0, (1.0 - new_w) / 2.0 - 1e-4))
        my = min(float(safe_margin), max(0.0, (1.0 - new_h) / 2.0 - 1e-4))
        boxes[idx, 2] = new_w
        boxes[idx, 3] = new_h
        boxes[idx, 0] = np.clip(cx, mx + half_w, 1.0 - mx - half_w)
        boxes[idx, 1] = np.clip(cy, my + half_h, 1.0 - my - half_h)
    return boxes.astype(np.float32)


def _fit_box_to_exact_aspect(
    cx: float,
    cy: float,
    w: float,
    h: float,
    aspect: float,
    min_w: float,
    max_w: float,
    min_h: float,
    max_h: float,
    safe_margin: float,
):
    """Resize a box to an exact aspect ratio using uniform scaling only.

    Important: unlike the old implementation, this function never independently
    clips width and height, so it preserves the requested aspect exactly.
    If the role bounds and the aspect ratio are mutually incompatible, we prefer
    exact aspect preservation and fit the box into the valid canvas / max-size
    bounds using the largest feasible uniform scale.
    """
    aspect = float(max(aspect, 1e-6))
    area = max(float(w) * float(h), 1e-6)
    base_w = math.sqrt(area * aspect)
    base_h = math.sqrt(area / aspect)

    canvas_max_w = max(1e-4, 1.0 - 2.0 * float(safe_margin))
    canvas_max_h = max(1e-4, 1.0 - 2.0 * float(safe_margin))
    eff_max_w = min(float(max_w), canvas_max_w)
    eff_max_h = min(float(max_h), canvas_max_h)

    s_upper = min(eff_max_w / max(base_w, 1e-8), eff_max_h / max(base_h, 1e-8))
    s_lower = max(float(min_w) / max(base_w, 1e-8), float(min_h) / max(base_h, 1e-8), 1e-8)

    if s_lower <= s_upper:
        if 1.0 < s_lower:
            s = s_lower
        elif 1.0 > s_upper:
            s = s_upper
        else:
            s = 1.0
    else:
        # Impossible to satisfy both min/max while keeping exact aspect.
        # Preserve aspect exactly and fit into the largest feasible size.
        s = max(s_upper, 1e-8)

    new_w = float(base_w * s)
    new_h = float(base_h * s)

    half_w = new_w / 2.0
    half_h = new_h / 2.0
    mx = min(float(safe_margin), max(0.0, (1.0 - new_w) / 2.0 - 1e-4))
    my = min(float(safe_margin), max(0.0, (1.0 - new_h) / 2.0 - 1e-4))
    new_cx = float(np.clip(cx, mx + half_w, 1.0 - mx - half_w))
    new_cy = float(np.clip(cy, my + half_h, 1.0 - my - half_h))
    return new_cx, new_cy, new_w, new_h


def enforce_svg_aspect_ratios(
    boxes: np.ndarray,
    labels: np.ndarray,
    svg_infos,
    svg_roles=None,
    preserve_svg_aspect: bool = True,
    safe_margin: float = 0.065,
):
    if (not preserve_svg_aspect) or not svg_infos:
        return boxes
    boxes = boxes.copy()
    svg_roles = svg_roles or {}
    svg_idxs = np.where(labels == 0)[0].tolist()
    if len(svg_idxs) == 0:
        return boxes
    for local_i, idx in enumerate(svg_idxs):
        aspect = svg_infos[local_i % len(svg_infos)].get('aspect', None)
        if aspect is None or aspect <= 0:
            continue
        aspect = float(np.clip(float(aspect), 0.05, 20.0))
        role = svg_roles.get(idx, SVG_ROLE_TEXTURE)
        (min_w, max_w), (min_h, max_h) = get_svg_size_range(role)
        cx, cy, w, h = [float(v) for v in boxes[idx]]
        cx, cy, new_w, new_h = _fit_box_to_exact_aspect(
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            aspect=aspect,
            min_w=min_w,
            max_w=max_w,
            min_h=min_h,
            max_h=max_h,
            safe_margin=safe_margin,
        )
        boxes[idx, 0] = cx
        boxes[idx, 1] = cy
        boxes[idx, 2] = new_w
        boxes[idx, 3] = new_h
    return boxes.astype(np.float32)


def apply_fixed_svg_size(
    boxes: np.ndarray,
    labels: np.ndarray,
    svg_infos,
    svg_roles=None,
    fixed_svg_long_side: float = 0.0,
    preserve_svg_aspect: bool = True,
    safe_margin: float = 0.065,
):
    """Force every SVG box to a fixed size while preserving aspect ratio.

    Strategy:
    - If aspect >= 1, width = fixed_svg_long_side and height = width / aspect
    - If aspect < 1, height = fixed_svg_long_side and width = height * aspect
    - Then fit the box into the canvas using a single uniform scale only.
    This guarantees the final SVG box is fixed-size in the long-side sense and
    still keeps the asset aspect ratio exactly.
    """
    if fixed_svg_long_side is None or float(fixed_svg_long_side) <= 0.0:
        return boxes
    if not svg_infos:
        return boxes
    boxes = boxes.copy()
    svg_roles = svg_roles or {}
    svg_idxs = np.where(labels == 0)[0].tolist()
    if len(svg_idxs) == 0:
        return boxes
    target_long = float(fixed_svg_long_side)
    for local_i, idx in enumerate(svg_idxs):
        aspect = svg_infos[local_i % len(svg_infos)].get('aspect', None)
        if aspect is None or aspect <= 0:
            aspect = max(float(boxes[idx, 2]) / max(float(boxes[idx, 3]), 1e-6), 1e-6)
        aspect = float(np.clip(float(aspect), 0.02, 50.0))
        role = svg_roles.get(idx, SVG_ROLE_TEXTURE)
        (min_w, max_w), (min_h, max_h) = get_svg_size_range(role, aspect=aspect, strict_asset_aspect=True)
        if preserve_svg_aspect:
            if aspect >= 1.0:
                base_w = target_long
                base_h = target_long / aspect
            else:
                base_h = target_long
                base_w = target_long * aspect
        else:
            base_w = target_long
            base_h = target_long
        cx, cy = float(boxes[idx, 0]), float(boxes[idx, 1])
        cx, cy, new_w, new_h = _fit_box_to_exact_aspect(
            cx=cx,
            cy=cy,
            w=float(base_w),
            h=float(base_h),
            aspect=aspect if preserve_svg_aspect else max(base_w / max(base_h, 1e-6), 1e-6),
            min_w=min_w,
            max_w=max_w,
            min_h=min_h,
            max_h=max_h,
            safe_margin=safe_margin,
        )
        boxes[idx, 0] = cx
        boxes[idx, 1] = cy
        boxes[idx, 2] = new_w
        boxes[idx, 3] = new_h
    return boxes.astype(np.float32)


def map_detected_face_to_image_box(img_box, face_norm):
    icx, icy, iw, ih = [float(v) for v in img_box]
    fc, fr, fw, fh = [float(v) for v in face_norm]
    img_x1 = icx - iw / 2.0
    img_y1 = icy - ih / 2.0
    fcx = img_x1 + fc * iw
    fcy = img_y1 + fr * ih
    face = np.array([fcx, fcy, fw * iw, fh * ih], dtype=np.float32)
    img_xyxy = box_xyxy(img_box)
    face[2] = min(float(face[2]), max(0.02, float(img_xyxy[2] - img_xyxy[0]) * 0.96))
    face[3] = min(float(face[3]), max(0.02, float(img_xyxy[3] - img_xyxy[1]) * 0.96))
    fx1, fy1, fx2, fy2 = box_xyxy(face)
    if fx1 < img_xyxy[0]:
        face[0] += img_xyxy[0] - fx1
    if fy1 < img_xyxy[1]:
        face[1] += img_xyxy[1] - fy1
    if fx2 > img_xyxy[2]:
        face[0] -= fx2 - img_xyxy[2]
    if fy2 > img_xyxy[3]:
        face[1] -= fy2 - img_xyxy[3]
    return face.astype(np.float32)


def append_faces_multi(boxes: np.ndarray, labels: np.ndarray, add_face: bool, image_infos):
    if not add_face or not image_infos:
        return boxes, labels, {}
    image_idxs = np.where(labels == 2)[0].tolist()
    if len(image_idxs) == 0:
        return boxes, labels, {}
    face_boxes = []
    face_owner = {}
    for local_i, img_idx in enumerate(image_idxs):
        info = image_infos[local_i % len(image_infos)]
        for face_norm in info.get("faces", []):
            face_boxes.append(map_detected_face_to_image_box(boxes[img_idx], face_norm))
    if len(face_boxes) == 0:
        return boxes, labels, {}
    new_boxes = np.concatenate([boxes, np.asarray(face_boxes, dtype=np.float32)], axis=0)
    new_labels = np.concatenate([labels, np.full((len(face_boxes),), 5, dtype=np.int64)], axis=0)
    cursor = len(boxes)
    for local_i, img_idx in enumerate(image_idxs):
        info = image_infos[local_i % len(image_infos)]
        for _ in info.get("faces", []):
            face_owner[cursor] = int(img_idx)
            cursor += 1
    return new_boxes, new_labels, face_owner


def keep_faces_inside_assigned_images(boxes: np.ndarray, labels: np.ndarray, face_owner: dict):
    boxes = boxes.copy()
    for fi, img_idx in face_owner.items():
        if fi >= len(boxes) or img_idx >= len(boxes):
            continue
        img_x1, img_y1, img_x2, img_y2 = box_xyxy(boxes[img_idx])
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


def fix_face_conflicts(boxes: np.ndarray, labels: np.ndarray, face_owner: dict, svg_roles=None, safe_margin: float = 0.065, steps: int = 25):
    boxes = keep_faces_inside_assigned_images(boxes, labels, face_owner)
    boxes = boxes.copy()
    face_idxs = sorted(face_owner.keys())
    for _ in range(int(steps)):
        changed = False
        for fi in face_idxs:
            img_idx = face_owner.get(fi, None)
            for j, lb in enumerate(labels):
                if j == fi or j == img_idx or int(lb) == 2:
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
        boxes = keep_faces_inside_assigned_images(boxes, labels, face_owner)
        if not changed:
            break
    return boxes


def style_position_penalty(boxes: np.ndarray, style: str) -> float:
    x1, y1, x2, y2 = union_bbox(boxes)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    union_w = x2 - x1
    union_h = y2 - y1
    penalty = 0.0
    if style == "right":
        penalty += abs(cx - 0.64) * 1.8
        penalty += abs(cy - 0.50) * 1.2
        penalty += max(0.0, union_w - 0.46) * 3.0
        penalty += max(0.0, union_h - 0.62) * 2.0
        penalty += max(0.0, 0.07 - x1) * 4.0
        penalty += max(0.0, 0.07 - y1) * 4.0
        penalty += max(0.0, 0.07 - (1.0 - x2)) * 4.0
        penalty += max(0.0, 0.07 - (1.0 - y2)) * 4.0
    elif style == "top":
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


def face_inside_image_penalty(boxes: np.ndarray, labels: np.ndarray, face_owner: dict) -> float:
    if not face_owner:
        return 0.0
    pen = 0.0
    for fi, img_idx in face_owner.items():
        if fi >= len(boxes) or img_idx >= len(boxes):
            pen += 100.0
            continue
        img = box_xyxy(boxes[img_idx])
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


def svg_role_penalty(boxes: np.ndarray, labels: np.ndarray, svg_roles=None) -> float:
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
    fg_idxs = np.where(labels != 5)[0].tolist()
    if len(fg_idxs) <= 2:
        return 0.0
    centers_x = boxes[fg_idxs, 0]
    centers_y = boxes[fg_idxs, 1]
    lefts = boxes[fg_idxs, 0] - boxes[fg_idxs, 2] / 2.0
    rights = boxes[fg_idxs, 0] + boxes[fg_idxs, 2] / 2.0
    tops = boxes[fg_idxs, 1] - boxes[fg_idxs, 3] / 2.0
    bottoms = boxes[fg_idxs, 1] + boxes[fg_idxs, 3] / 2.0
    candidates = [
        float(np.std(centers_x)), float(np.std(centers_y)),
        float(np.std(lefts)), float(np.std(rights)),
        float(np.std(tops)), float(np.std(bottoms)),
    ]
    return float(min(candidates))


def whitespace_proxy_score(boxes: np.ndarray, labels: np.ndarray) -> float:
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
    area = total_fg_area(fg)
    WR_proxy = float(np.clip(1.0 - area, 0.0, 1.0))
    union_w = max(1e-6, x2 - x1)
    union_h = max(1e-6, y2 - y1)
    scatter_pen = max(0.0, union_w - 0.58) + max(0.0, union_h - 0.62)
    D_conn_proxy = float(np.clip(1.0 - 0.8 * scatter_pen, 0.0, 1.0))
    n_fg = len(fg)
    frag_proxy = 1.0 / (1.0 + 0.06 * max(0, n_fg - 3))
    score = WR_proxy * D_conn_proxy * frag_proxy * S_style
    return float(np.clip(score, 0.0, 1.0))


def score_layout(
    boxes: np.ndarray,
    labels: np.ndarray,
    style: str,
    max_fg_area: float,
    svg_roles=None,
    face_owner=None,
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
    face_in_pen = face_inside_image_penalty(boxes, labels, face_owner or {})
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
        "face_owner": {int(k): int(v) for k, v in (face_owner or {}).items()},
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


def choose_labels(style: str, max_elems: int, add_face: bool):
    pool = STYLE_LABEL_POOLS[style]
    gen_cap = max_elems - 1 if add_face else max_elems
    if gen_cap <= 0:
        raise ValueError("max_elems must be >= 1, and >= 2 if add_face is enabled.")
    valid = [seq for seq in pool if len(seq) <= gen_cap and 2 in seq]
    if len(valid) == 0:
        raise ValueError(f"No valid label preset for style={style}, max_elems={max_elems}, add_face={add_face}")
    return np.array(random.choice(valid), dtype=np.int64)


def count_expected_appended_faces(labels, image_infos, add_face: bool):
    if not add_face or not image_infos:
        return 0
    image_count = sum(1 for x in labels if x == 2)
    total = 0
    for i in range(image_count):
        total += len(image_infos[i % len(image_infos)].get("faces", []))
    return int(total)


def validate_custom_labels(labels, max_elems, add_face, image_infos, svg_infos):
    if len(labels) == 0:
        raise ValueError("Custom labels cannot be empty.")
    bad = [x for x in labels if x not in [0, 1, 2]]
    if len(bad) > 0:
        raise ValueError(f"Custom generation labels can only contain 0/1/2. Bad labels: {bad}")
    expected_faces = count_expected_appended_faces(labels, image_infos, bool(add_face))
    total_visible = len(labels) + expected_faces
    if total_visible > max_elems:
        raise ValueError(
            f"Visible element count would be {total_visible} ({len(labels)} labels + {expected_faces} appended faces), "
            f"but max_elems={max_elems}. Increase --max_elems or reduce --max_detected_faces."
        )
    if image_infos and 2 not in labels:
        raise ValueError("When --image_paths is used, labels must include at least one image (label 2).")
    if svg_infos and 0 not in labels:
        raise ValueError("When --svg_paths is used, labels must include at least one SVG (label 0).")


def resolve_infer_mode(infer_mode: str, requested_k: int):
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
        "w_ov": 35.0,
        "w_area": 8.0,
        "w_face_conf": 100.0,
        "w_face_in": 8.0,
        "w_pos": 0.8,
        "w_svg": 4.0,
        "w_frame": 1.0,
        "w_white": 24.0,
        "w_margin": 18.0,
        "w_align": 0.8,
    }


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
    image_infos,
    svg_infos,
    preserve_image_aspect: bool,
    preserve_svg_aspect: bool,
    svg_role_affects_size: bool,
    fixed_svg_long_side: float,
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
        svg_roles = sample_svg_roles(labels=labels_gen, rng=rng, svg_small_prob=svg_small_prob)
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
            svg_role_affects_size=svg_role_affects_size,
        )
        boxes = enforce_image_aspect_ratios(
            boxes,
            labels=labels_gen,
            image_infos=image_infos,
            preserve_image_aspect=preserve_image_aspect,
            safe_margin=safe_margin,
        )
        boxes = enforce_svg_aspect_ratios(
            boxes,
            labels=labels_gen,
            svg_infos=svg_infos,
            svg_roles=svg_roles,
            preserve_svg_aspect=preserve_svg_aspect,
            safe_margin=safe_margin,
        )
        if mode_cfg["use_layout_repair"]:
            boxes = clip_boxes_xywh_margin(boxes, labels_gen, svg_roles=svg_roles, safe_margin=safe_margin)
        else:
            boxes = clip_boxes_xywh(boxes, labels_gen, svg_roles=svg_roles)
        cand_boxes, cand_labels, face_owner = append_faces_multi(
            boxes,
            labels_gen,
            add_face=add_face,
            image_infos=image_infos,
        )
        if mode_cfg["use_layout_repair"]:
            cand_boxes = clip_boxes_xywh_margin(cand_boxes, cand_labels, svg_roles=svg_roles, safe_margin=safe_margin)
            cand_boxes = keep_faces_inside_assigned_images(cand_boxes, cand_labels, face_owner)
            cand_boxes = fix_face_conflicts(cand_boxes, cand_labels, face_owner, svg_roles=svg_roles, safe_margin=safe_margin, steps=25)
            cand_boxes = repel_overlaps(cand_boxes, cand_labels, gap_text=gap_text, gap_other=gap_other, steps=20, safe_margin=safe_margin, svg_roles=svg_roles)
            cand_boxes = keep_faces_inside_assigned_images(cand_boxes, cand_labels, face_owner)
        else:
            cand_boxes = clip_boxes_xywh(cand_boxes, cand_labels, svg_roles=svg_roles)

        # Final strict pass: restore exact SVG aspect ratios after all repair steps.
        cand_boxes = enforce_svg_aspect_ratios(
            cand_boxes,
            labels=cand_labels,
            svg_infos=svg_infos,
            svg_roles=svg_roles,
            preserve_svg_aspect=preserve_svg_aspect,
            safe_margin=safe_margin,
        )
        cand_boxes = apply_fixed_svg_size(
            cand_boxes,
            labels=cand_labels,
            svg_infos=svg_infos,
            svg_roles=svg_roles,
            fixed_svg_long_side=fixed_svg_long_side,
            preserve_svg_aspect=preserve_svg_aspect,
            safe_margin=safe_margin,
        )
        score, meta = score_layout(
            cand_boxes,
            labels=cand_labels,
            style=style,
            max_fg_area=max_fg_area,
            svg_roles=svg_roles,
            face_owner=face_owner,
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


def compute_layout_face_cover_stats(boxes: np.ndarray, labels: np.ndarray):
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


def aggregate_cover_summary(results):
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
    return {
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


def format_cover_summary_block(name: str, stats: dict):
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


def diff_cover_summary(curr: dict, base: dict):
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


def write_cover_summary_files(out_dir: Path, args, results):
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


def write_metrics_csv(out_dir: Path, metas, labels_list, results):
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
        'num_faces', 'svg_roles', 'face_owner',
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
                'face_owner': str(meta.get('face_owner', {})),
            }
            writer.writerow(row)


def _as_plain_dict(obj):
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj).copy()
    return {}


def _strip_state_dict_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    prefixes = ["module.", "netG.", "generator.", "model."]
    out = {}
    for k, v in state_dict.items():
        nk = str(k)
        changed = True
        while changed:
            changed = False
            for pref in prefixes:
                if nk.startswith(pref):
                    nk = nk[len(pref):]
                    changed = True
        out[nk] = v
    return out


def _extract_netG_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    candidate_keys = ["netG", "generator", "model_state_dict", "state_dict", "model", "net"]
    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return _strip_state_dict_prefix(ckpt[key])
    tensor_like = [hasattr(v, "shape") for v in ckpt.values()]
    if len(tensor_like) > 0 and all(tensor_like):
        return _strip_state_dict_prefix(ckpt)
    raise KeyError(
        "Cannot find generator weights in checkpoint. Expected one of: "
        "netG, generator, model_state_dict, state_dict, model, net, or a raw state_dict. "
        f"Actual keys: {list(ckpt.keys())[:30]}"
    )


def _infer_generator_hparams_from_state_dict(state_dict: dict) -> dict:
    inferred = {}
    if "fc_z.weight" in state_dict:
        inferred["G_d_model"] = int(state_dict["fc_z.weight"].shape[0])
        inferred["latent_size"] = int(state_dict["fc_z.weight"].shape[1])
    if "emb_label.weight" in state_dict:
        inferred["num_label_from_ckpt"] = int(state_dict["emb_label.weight"].shape[0])
        inferred.setdefault("G_d_model", int(state_dict["emb_label.weight"].shape[1]))
    layer_ids = []
    for k in state_dict.keys():
        k = str(k)
        if k.startswith("transformer.layers."):
            parts = k.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                layer_ids.append(int(parts[2]))
    if layer_ids:
        inferred["G_num_layers"] = max(layer_ids) + 1
    d_model = inferred.get("G_d_model", 256)
    inferred["G_nhead"] = 4 if d_model % 4 == 0 else 1
    return inferred


def _get_train_arg(train_args: dict, key: str, default=None):
    if isinstance(train_args, dict):
        return train_args.get(key, default)
    return getattr(train_args, key, default)


def load_generator_checkpoint_for_inference(ckpt_path: str, device, cli_args):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_netG_state_dict(ckpt)
    if isinstance(ckpt, dict) and "args" in ckpt:
        train_args = _as_plain_dict(ckpt["args"])
        print(f"[CKPT] Found training args in checkpoint: {ckpt_path}")
    else:
        print(f"[CKPT][WARN] No 'args' found in checkpoint: {ckpt_path}")
        print("[CKPT][WARN] Using command-line/default args and inferring model shape from weights.")
        train_args = {}
    inferred = _infer_generator_hparams_from_state_dict(state_dict)
    defaults = {
        "dataset": "crello",
        "latent_size": 4,
        "G_d_model": 256,
        "G_nhead": 4,
        "G_num_layers": 4,
    }
    merged = {}
    merged.update(defaults)
    merged.update(train_args)
    merged.update({k: v for k, v in inferred.items() if k != "num_label_from_ckpt"})
    print("[CKPT] Loaded checkpoint:", ckpt_path)
    print("[CKPT] State dict keys:", len(state_dict))
    print(
        "[CKPT] Inference hparams:",
        {
            "dataset": merged.get("dataset"),
            "latent_size": merged.get("latent_size"),
            "G_d_model": merged.get("G_d_model"),
            "G_nhead": merged.get("G_nhead"),
            "G_num_layers": merged.get("G_num_layers"),
            "num_label_from_ckpt": inferred.get("num_label_from_ckpt"),
        },
    )
    return merged, state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--style", type=str, choices=["right", "hybrid", "top", "frame"], required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_fg_area", type=float, default=0.33)
    parser.add_argument("--max_elems", type=int, default=6)
    parser.add_argument("--add_face", type=int, default=1)
    parser.add_argument("--image_paths", type=str, nargs="*", default=None,
                        help="多張使用者圖片路徑，依序對應到 layout 裡的 label=2 image box")
    parser.add_argument("--svg_paths", type=str, nargs="*", default=None,
                        help="多個 SVG 路徑，依序對應到 layout 裡的 label=0 SVG box；只用來讀取 SVG 長寬比")
    parser.add_argument("--face_model", type=str, default="yolov8n-face.pt")
    parser.add_argument("--face_conf", type=float, default=0.3)
    parser.add_argument("--max_detected_faces", type=int, default=1,
                        help="每張輸入圖片最多保留幾張偵測到的人臉")
    parser.add_argument("--preserve_image_aspect", type=int, default=1,
                        help="1=依照每張原圖 W/H 修正對應的 label=2 image box")
    parser.add_argument("--preserve_svg_aspect", type=int, default=1,
                        help="1=依照每個 SVG 的長寬比修正對應的 label=0 SVG box")
    parser.add_argument("--svg_role_affects_size", type=int, default=0,
                        help="1=保留 icon/texture 對 SVG 尺寸的影響；0=只保留位置 prior，不改 SVG box 尺寸")
    parser.add_argument("--fixed_svg_long_side", type=float, default=0.18,
                        help=">0 時，強制每個 label=0 SVG box 的長邊固定為這個 normalized 大小，例如 0.18")
    parser.add_argument("--preview_w", type=int, default=1200)
    parser.add_argument("--preview_h", type=int, default=800)
    parser.add_argument("--preview_border_width", type=int, default=1,
                        help="layout preview 的框線粗細。1 會最接近 Crello 標註式空心框")
    parser.add_argument("--preview_fill_alpha", type=int, default=0,
                        help="保留參數相容性；本版本固定空心框，不填色")
    parser.add_argument("--svg_small_prob", type=float, default=0.72,
                        help="single SVG 被視為小 icon 的機率")
    parser.add_argument("--infer_mode", type=str, choices=["raw", "final"], default="final")
    parser.add_argument("--labels", type=int, nargs="*", default=None,
                        help="固定元素，例如 --labels 2 2 0 0")
    parser.add_argument("--frame_band_margin", type=float, default=0.10)
    parser.add_argument("--frame_target_w", type=float, default=0.28)
    parser.add_argument("--frame_target_h", type=float, default=0.38)
    parser.add_argument("--frame_max_ar", type=float, default=1.8)
    parser.add_argument("--safe_margin", type=float, default=0.065)
    parser.add_argument("--gap_text", type=float, default=0.035)
    parser.add_argument("--gap_other", type=float, default=0.025)
    parser.add_argument("--summary_name", type=str, default="")
    parser.add_argument("--compare_stats_json", type=str, default="")
    parser.add_argument("--compare_name", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_args, state_dict = load_generator_checkpoint_for_inference(args.resume_ckpt, device=device, cli_args=args)
    dataset = get_dataset(_get_train_arg(train_args, "dataset", "crello"), "val", T.Compose([]))
    num_label = dataset.num_classes
    ckpt_num_label = _infer_generator_hparams_from_state_dict(state_dict).get("num_label_from_ckpt")
    if ckpt_num_label is not None and int(ckpt_num_label) != int(num_label):
        print(
            f"[CKPT][WARN] dataset.num_classes={num_label}, but ckpt emb_label has {ckpt_num_label}. "
            "Using ckpt label count for model construction."
        )
        num_label = int(ckpt_num_label)
    netG = Generator(
        int(_get_train_arg(train_args, "latent_size", 4)),
        num_label,
        d_model=int(_get_train_arg(train_args, "G_d_model", 256)),
        nhead=int(_get_train_arg(train_args, "G_nhead", 4)),
        num_layers=int(_get_train_arg(train_args, "G_num_layers", 4)),
    ).eval().requires_grad_(False).to(device)
    missing, unexpected = netG.load_state_dict(state_dict, strict=False)
    if missing:
        print("[CKPT][WARN] Missing keys when loading netG:", missing[:20], "..." if len(missing) > 20 else "")
    if unexpected:
        print("[CKPT][WARN] Unexpected keys when loading netG:", unexpected[:20], "..." if len(unexpected) > 20 else "")
    if len(missing) > 0 or len(unexpected) > 0:
        print("[CKPT][WARN] Checkpoint loaded with strict=False. If outputs look wrong, this ckpt may be a different architecture.")

    image_paths = [str(p) for p in (args.image_paths or []) if str(p).strip()]
    svg_paths = [str(p) for p in (args.svg_paths or []) if str(p).strip()]
    for p in image_paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Image path not found: {p}")
    for p in svg_paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"SVG path not found: {p}")
    if image_paths and not Path(args.face_model).is_file():
        raise FileNotFoundError(f"Face model not found: {args.face_model}")

    image_infos = []
    if image_paths:
        image_infos = collect_image_inputs(
            image_paths=image_paths,
            face_model=args.face_model,
            face_conf=args.face_conf,
            max_detected_faces=args.max_detected_faces,
        )
        print("[IMAGES] num input images:", len(image_infos))
        for idx, info in enumerate(image_infos):
            print(f"[IMAGES] #{idx}: path={info['path']}")
            print(f"[IMAGES] #{idx}: aspect={info['aspect']}")
            print(f"[IMAGES] #{idx}: detected_faces={len(info['faces'])}")
            print(f"[IMAGES] #{idx}: face_boxes={info['faces']}")

    svg_infos = []
    if svg_paths:
        svg_infos = collect_svg_inputs(svg_paths)
        print("[SVGS] num input svgs:", len(svg_infos))
        for idx, info in enumerate(svg_infos):
            print(f"[SVGS] #{idx}: path={info['path']}")
            print(f"[SVGS] #{idx}: aspect={info['aspect']}")

    results = []
    metas = []
    labels_list = []
    for i in tqdm(range(args.n), ncols=100):
        if args.labels is not None:
            validate_custom_labels(args.labels, args.max_elems, bool(args.add_face), image_infos, svg_infos)
            labels_gen = np.array(args.labels, dtype=np.int64)
        else:
            labels_gen = choose_labels(args.style, args.max_elems, bool(args.add_face))
        boxes, labels, meta = sample_best_of_k(
            netG=netG,
            labels_gen=labels_gen,
            latent_size=int(_get_train_arg(train_args, "latent_size", 4)),
            k=args.k,
            device=device,
            style=args.style,
            max_fg_area=args.max_fg_area,
            add_face=bool(args.add_face),
            image_infos=image_infos,
            svg_infos=svg_infos,
            preserve_image_aspect=bool(args.preserve_image_aspect),
            preserve_svg_aspect=bool(args.preserve_svg_aspect),
            svg_role_affects_size=bool(args.svg_role_affects_size),
            fixed_svg_long_side=float(args.fixed_svg_long_side),
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
        img = render_layout_preview_exact(
            boxes=boxes,
            labels=labels,
            colors=dataset.colors,
            preview_w=args.preview_w,
            preview_h=args.preview_h,
            border_width=args.preview_border_width,
            fill_alpha=args.preview_fill_alpha,
        )
        img.save(out_dir / f"{i:04d}.png")

    with open(out_dir / "generated_layouts.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(metas, f)
    write_metrics_csv(out_dir, metas, labels_list, results)
    write_cover_summary_files(out_dir, args, results)

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
    print("image_paths:", image_paths)
    print("svg_paths:", svg_paths)
    print("face_model:", args.face_model)
    print("face_conf:", args.face_conf)
    print("max_detected_faces:", args.max_detected_faces)
    print("preserve_image_aspect:", args.preserve_image_aspect)
    print("preserve_svg_aspect:", args.preserve_svg_aspect)
    print("svg_role_affects_size:", args.svg_role_affects_size)
    print("fixed_svg_long_side:", args.fixed_svg_long_side)
    print("preview_w:", args.preview_w)
    print("preview_h:", args.preview_h)
    print("preview_border_width:", args.preview_border_width)
    print("preview_fill_alpha:", args.preview_fill_alpha)
    print("svg_small_prob:", args.svg_small_prob)
    print("safe_margin:", args.safe_margin)
    print("gap_text:", args.gap_text)
    print("gap_other:", args.gap_other)
    if args.labels is not None:
        expected_faces = count_expected_appended_faces(args.labels, image_infos, bool(args.add_face))
        print("fixed generation labels:", args.labels)
        print("expected appended faces:", expected_faces)
        print("visible count =", len(args.labels) + expected_faces)
    else:
        print("label pool mode: enabled")


if __name__ == "__main__":
    main()
