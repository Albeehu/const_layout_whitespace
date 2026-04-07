#用來比較baseline跟new的ckpt推論哪個比較不會蓋臉
"""

"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import csv
import math
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T

from data import get_dataset
from model.layoutganpp import Generator
import infer_official_face_v2 as official


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Face metrics on FINAL inferred layouts
# -------------------------
def box_xyxy_np(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def area_xyxy_np(boxes_xyxy: np.ndarray) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=np.float32)
    w = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0, None)
    h = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0, None)
    return np.clip(w * h, 1e-8, None)


def pairwise_intersection_np(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    if len(a_xyxy) == 0 or len(b_xyxy) == 0:
        return np.zeros((len(a_xyxy), len(b_xyxy)), dtype=np.float32)
    x1 = np.maximum(a_xyxy[:, None, 0], b_xyxy[None, :, 0])
    y1 = np.maximum(a_xyxy[:, None, 1], b_xyxy[None, :, 1])
    x2 = np.minimum(a_xyxy[:, None, 2], b_xyxy[None, :, 2])
    y2 = np.minimum(a_xyxy[:, None, 3], b_xyxy[None, :, 3])
    iw = np.clip(x2 - x1, 0.0, None)
    ih = np.clip(y2 - y1, 0.0, None)
    return iw * ih


def pairwise_iou_np(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    if len(a_xyxy) == 0 or len(b_xyxy) == 0:
        return np.zeros((len(a_xyxy), len(b_xyxy)), dtype=np.float32)
    inter = pairwise_intersection_np(a_xyxy, b_xyxy)
    area_a = area_xyxy_np(a_xyxy)
    area_b = area_xyxy_np(b_xyxy)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def compute_face_metrics(
    boxes_xywh: np.ndarray,
    labels: np.ndarray,
    face_id: int = 5,
    container_ids: Tuple[int, ...] = (2, 3),
    contain_thresh: float = 0.95,
) -> Dict[str, float]:
    face_mask = labels == face_id
    other_mask = ~face_mask

    num_faces = int(face_mask.sum())
    if num_faces == 0:
        return {
            'num_faces': 0,
            'maxIoU': 0.0,
            'meanCov': 0.0,
            'maxCov': 0.0,
            'anyCover': 0.0,
            'face_cover_gt0_count': 0.0,
            'face_cover_gt005_count': 0.0,
            'face_cover_gt010_count': 0.0,
            'face_mean_coverage_sum': 0.0,
        }

    face_boxes = np.stack([box_xyxy_np(b) for b in boxes_xywh[face_mask]], axis=0)
    other_boxes_xywh = boxes_xywh[other_mask]
    other_labels = labels[other_mask]

    if len(other_boxes_xywh) == 0:
        return {
            'num_faces': num_faces,
            'maxIoU': 0.0,
            'meanCov': 0.0,
            'maxCov': 0.0,
            'anyCover': 0.0,
            'face_cover_gt0_count': 0.0,
            'face_cover_gt005_count': 0.0,
            'face_cover_gt010_count': 0.0,
            'face_mean_coverage_sum': 0.0,
        }

    other_boxes = np.stack([box_xyxy_np(b) for b in other_boxes_xywh], axis=0)
    ious = pairwise_iou_np(other_boxes, face_boxes)
    max_iou = float(ious.max()) if ious.size > 0 else 0.0

    inter = pairwise_intersection_np(other_boxes, face_boxes)
    face_area = area_xyxy_np(face_boxes)

    if len(other_labels) > 0:
        is_container = np.zeros((len(other_labels),), dtype=bool)
        for cid in container_ids:
            is_container |= (other_labels == cid)
        if is_container.any():
            contain_ratio = inter[is_container] / face_area[None, :]
            contain_mask = contain_ratio > contain_thresh
            inter = inter.copy()
            inter[is_container] = inter[is_container] * (~contain_mask)

    covered = inter.sum(axis=0)
    coverage = np.clip(covered / face_area, 0.0, 1.0)

    return {
        'num_faces': num_faces,
        'maxIoU': max_iou,
        'meanCov': float(coverage.mean()),
        'maxCov': float(coverage.max()),
        'anyCover': float((coverage > 0.0).any()),
        'face_cover_gt0_count': float((coverage > 0.0).sum()),
        'face_cover_gt005_count': float((coverage > 0.05).sum()),
        'face_cover_gt010_count': float((coverage > 0.10).sum()),
        'face_mean_coverage_sum': float(coverage.sum()),
    }


# -------------------------
# Model loading / generation
# -------------------------
def load_generator(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt['args']
    dataset = get_dataset(train_args['dataset'], 'val', T.Compose([]))
    num_label = dataset.num_classes

    netG = Generator(
        train_args['latent_size'],
        num_label,
        d_model=train_args['G_d_model'],
        nhead=train_args['G_nhead'],
        num_layers=train_args['G_num_layers'],
    ).eval().requires_grad_(False).to(device)

    netG.load_state_dict(ckpt['netG'])
    return netG, train_args, dataset


def build_label_sequences(args) -> List[np.ndarray]:
    seqs = []
    set_seed(args.seed)
    for _ in range(args.n):
        if args.labels is not None:
            official.validate_custom_labels(args.labels, args.max_elems, bool(args.add_face))
            labels_gen = np.array(args.labels, dtype=np.int64)
        else:
            labels_gen = official.choose_labels(args.style, args.max_elems, bool(args.add_face))
        seqs.append(labels_gen)
    return seqs


def run_one_model(ckpt_path: str, tag: str, args, label_sequences: List[np.ndarray], device: torch.device):
    set_seed(args.seed)
    netG, train_args, dataset = load_generator(ckpt_path, device)

    rows = []
    agg = {
        'num_layouts': 0,
        'num_faces': 0,
        'sum_maxIoU': 0.0,
        'sum_meanCov': 0.0,
        'sum_maxCov': 0.0,
        'sum_anyCover': 0.0,
        'face_cover_gt0_count': 0.0,
        'face_cover_gt005_count': 0.0,
        'face_cover_gt010_count': 0.0,
        'face_mean_coverage_sum': 0.0,
    }

    for i, labels_gen in enumerate(label_sequences):
        boxes, labels, meta = official.sample_best_of_k(
            netG=netG,
            labels_gen=labels_gen,
            latent_size=train_args['latent_size'],
            k=args.k,
            device=device,
            style=args.style,
            max_fg_area=args.max_fg_area,
            add_face=bool(args.add_face),
            sample_index=i,
            svg_small_prob=args.svg_small_prob,
        )

        m = compute_face_metrics(boxes, labels, contain_thresh=args.contain_thresh)
        rows.append({
            'idx': i,
            'tag': tag,
            'num_faces': m['num_faces'],
            'maxIoU': m['maxIoU'],
            'meanCov': m['meanCov'],
            'maxCov': m['maxCov'],
            'anyCover': m['anyCover'],
            'score': float(meta['score']),
            'overlap': float(meta['overlap']),
            'area': float(meta['area']),
            'face_inside_penalty': float(meta['face_inside_penalty']),
            'face_conflict_penalty': float(meta['face_conflict_penalty']),
            'svg_penalty': float(meta['svg_penalty']),
        })

        agg['num_layouts'] += 1
        agg['num_faces'] += m['num_faces']
        agg['sum_maxIoU'] += m['maxIoU']
        agg['sum_meanCov'] += m['meanCov']
        agg['sum_maxCov'] += m['maxCov']
        agg['sum_anyCover'] += m['anyCover']
        agg['face_cover_gt0_count'] += m['face_cover_gt0_count']
        agg['face_cover_gt005_count'] += m['face_cover_gt005_count']
        agg['face_cover_gt010_count'] += m['face_cover_gt010_count']
        agg['face_mean_coverage_sum'] += m['face_mean_coverage_sum']

    summary = summarize(tag, agg)
    return rows, summary


def summarize(tag: str, agg: Dict[str, float]) -> Dict[str, float]:
    n_layouts = max(int(agg['num_layouts']), 1)
    n_faces = max(int(agg['num_faces']), 1)
    return {
        'tag': tag,
        'num_layouts': int(agg['num_layouts']),
        'num_faces': int(agg['num_faces']),
        'layout_mean_maxIoU': agg['sum_maxIoU'] / n_layouts,
        'layout_mean_meanCov': agg['sum_meanCov'] / n_layouts,
        'layout_mean_maxCov': agg['sum_maxCov'] / n_layouts,
        'layout_anyCover_ratio': agg['sum_anyCover'] / n_layouts,
        'face_cover_gt0_ratio': agg['face_cover_gt0_count'] / n_faces,
        'face_cover_gt005_ratio': agg['face_cover_gt005_count'] / n_faces,
        'face_cover_gt010_ratio': agg['face_cover_gt010_count'] / n_faces,
        'face_mean_coverage': agg['face_mean_coverage_sum'] / n_faces,
    }


def write_summary(path: Path, summary_a: Dict[str, float], summary_b: Dict[str, float]):
    with open(path, 'w') as f:
        for s in [summary_a, summary_b]:
            f.write(f"=== {s['tag']} ===\n")
            f.write(f"num_layouts={s['num_layouts']} num_faces={s['num_faces']}\n")
            for k in [
                'layout_mean_maxIoU', 'layout_mean_meanCov', 'layout_mean_maxCov',
                'layout_anyCover_ratio', 'face_cover_gt0_ratio', 'face_cover_gt005_ratio',
                'face_cover_gt010_ratio', 'face_mean_coverage'
            ]:
                f.write(f"{k}={s[k]:.4f}\n")
            f.write('\n')

        f.write(f"=== {summary_a['tag']} minus {summary_b['tag']} ===\n")
        for k in [
            'layout_mean_maxIoU', 'layout_mean_meanCov', 'layout_mean_maxCov',
            'layout_anyCover_ratio', 'face_cover_gt0_ratio', 'face_cover_gt005_ratio',
            'face_cover_gt010_ratio', 'face_mean_coverage'
        ]:
            diff = summary_a[k] - summary_b[k]
            f.write(f"{k}: {diff}\n")


def print_summary(summary: Dict[str, float]):
    print(f"=== {summary['tag']} ===")
    print(f"num_layouts={summary['num_layouts']} num_faces={summary['num_faces']}")
    for k in [
        'layout_mean_maxIoU', 'layout_mean_meanCov', 'layout_mean_maxCov',
        'layout_anyCover_ratio', 'face_cover_gt0_ratio', 'face_cover_gt005_ratio',
        'face_cover_gt010_ratio', 'face_mean_coverage'
    ]:
        print(f"{k}={summary[k]:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare two checkpoints using OFFICIAL inference pipeline.')
    parser.add_argument('--ckpt_a', type=str, required=True)
    parser.add_argument('--ckpt_b', type=str, required=True)
    parser.add_argument('--tag_a', type=str, default='improved')
    parser.add_argument('--tag_b', type=str, default='baseline')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--style', type=str, choices=['right', 'hybrid', 'top'], required=True)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max_fg_area', type=float, default=0.33)
    parser.add_argument('--max_elems', type=int, default=5)
    parser.add_argument('--add_face', type=int, default=1)
    parser.add_argument('--svg_small_prob', type=float, default=0.72)
    parser.add_argument('--labels', type=int, nargs='*', default=None)
    parser.add_argument('--contain_thresh', type=float, default=0.95)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    label_sequences = build_label_sequences(args)

    rows_a, summary_a = run_one_model(args.ckpt_a, args.tag_a, args, label_sequences, device)
    rows_b, summary_b = run_one_model(args.ckpt_b, args.tag_b, args, label_sequences, device)

    print_summary(summary_a)
    print_summary(summary_b)

    print(f"=== {summary_a['tag']} minus {summary_b['tag']} ===")
    for k in [
        'layout_mean_maxIoU', 'layout_mean_meanCov', 'layout_mean_maxCov',
        'layout_anyCover_ratio', 'face_cover_gt0_ratio', 'face_cover_gt005_ratio',
        'face_cover_gt010_ratio', 'face_mean_coverage'
    ]:
        print(f"{k}: {summary_a[k] - summary_b[k]}")

    csv_path = out_dir / 'per_layout_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = list(rows_a[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_a + rows_b:
            writer.writerow(r)

    summary_path = out_dir / 'summary.txt'
    write_summary(summary_path, summary_a, summary_b)
    print(f"[write] {csv_path}")
    print(f"[write] {summary_path}")


if __name__ == '__main__':
    main()
