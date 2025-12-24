#debug
import os
import csv
import argparse
import random
from typing import Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.utils import to_dense_batch

from data import get_dataset
from model.layoutganpp import Generator

# ====== label ids (你提供的 mapping) ======
FACE_ID = 5
IMAGE_ID = 2
BG_ID = 3
CONTAINER_IDS = {IMAGE_ID, BG_ID}


# ------------------------------
# utils: reproducibility
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 盡可能 deterministic（有些 op 仍可能非完全 deterministic）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# box helpers
# ------------------------------
def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] -> [x1, y1, x2, y2]"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area_xyxy(xyxy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
    h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
    return (w * h).clamp(min=eps)


def box_intersection_area_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (K,4), b: (M,4) -> inter area: (K,M)
    """
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    return inter_w * inter_h


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU matrix: (N,M)"""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ------------------------------
# metrics aligned with your training intention
# ------------------------------
@torch.no_grad()
def compute_metrics_for_one_layout(
    boxes_xywh: torch.Tensor,
    labels: torch.Tensor,
    face_id: int = FACE_ID,
    container_ids: set = CONTAINER_IDS,
    contain_thresh: float = 0.95,
) -> Dict[str, float]:
    """
    回傳這張 layout 的：
      - num_face
      - num_nonface
      - num_overlapped_iou_pos (IoU>0)
      - max_iou_nonface_to_face
      - mean_coverage (per-face)
      - max_coverage (per-face)

    coverage(face) = sum_other inter_area(other, face) / area(face)
    並做容器免罰：容器(2/3) 若幾乎包含 face（inter/face_area > thresh）則該 pair inter 設 0
    """
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    face_mask = (labels == face_id)
    other_mask = ~face_mask

    face_boxes = boxes_xyxy[face_mask]
    other_boxes = boxes_xyxy[other_mask]
    other_labels = labels[other_mask]

    num_face = int(face_mask.sum().item())
    num_nonface = int(other_mask.sum().item())

    if num_face == 0 or num_nonface == 0:
        return {
            "num_face": float(num_face),
            "num_nonface": float(num_nonface),
            "num_overlapped_iou_pos": 0.0,
            "max_iou": 0.0,
            "mean_coverage": 0.0,
            "max_coverage": 0.0,
        }

    # IoU (debug friendly)
    ious = box_iou_xyxy(other_boxes, face_boxes)  # (K,M)
    max_iou_per_other = ious.max(dim=1).values
    num_overlapped = float((max_iou_per_other > 0.0).sum().item())
    max_iou = float(ious.max().item())

    # Coverage (training-aligned)
    inter = box_intersection_area_xyxy(other_boxes, face_boxes)  # (K,M)
    face_area = box_area_xyxy(face_boxes)  # (M,)

    # container exemption: if container nearly contains face, ignore that pair
    is_container = torch.zeros_like(other_labels, dtype=torch.bool)
    for cid in container_ids:
        is_container |= (other_labels == cid)

    if is_container.any():
        inter_c = inter[is_container]  # (Kc,M)
        contain_ratio = inter_c / face_area[None, :]
        contain_mask = (contain_ratio > contain_thresh)
        inter_c = inter_c * (~contain_mask).to(inter.dtype)
        inter = inter.clone()
        inter[is_container] = inter_c

    covered = inter.sum(dim=0)  # (M,)
    coverage = (covered / face_area).clamp(0.0, 1.0)  # (M,)

    mean_cov = float(coverage.mean().item())
    max_cov = float(coverage.max().item())

    return {
        "num_face": float(num_face),
        "num_nonface": float(num_nonface),
        "num_overlapped_iou_pos": float(num_overlapped),
        "max_iou": float(max_iou),
        "mean_coverage": float(mean_cov),
        "max_coverage": float(max_cov),
    }


# ------------------------------
# visualization
# ------------------------------
def draw_layout(
    ax,
    boxes_xywh: torch.Tensor,
    labels: torch.Tensor,
    title: str,
    face_id: int = FACE_ID,
    iou_threshold_green: float = 0.0,
):
    """
    顏色規則：
      - face: 紅色
      - 非face 且 maxIoU(face)>threshold: 綠色
      - 其他: 藍色
    """
    boxes_xywh = boxes_xywh.cpu()
    labels = labels.cpu()

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    face_mask = labels == face_id
    other_mask = ~face_mask

    face_boxes = boxes_xyxy[face_mask]
    other_boxes = boxes_xyxy[other_mask]

    overlapped_idx = set()
    if face_boxes.numel() > 0 and other_boxes.numel() > 0:
        ious = box_iou_xyxy(other_boxes, face_boxes)
        max_iou_per_other = ious.max(dim=1).values
        idx = (max_iou_per_other > iou_threshold_green).nonzero(as_tuple=False).view(-1).tolist()
        overlapped_idx = set(idx)

    # draw face
    for fb in face_boxes:
        x1, y1, x2, y2 = fb.tolist()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.5, edgecolor="red"))

    # draw others
    if other_boxes.numel() > 0:
        for idx, ob in enumerate(other_boxes):
            x1, y1, x2, y2 = ob.tolist()
            if idx in overlapped_idx:
                color, lw = "green", 2.0
            else:
                color, lw = "blue", 1.0
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=lw, edgecolor=color))

    ax.set_title(title, fontsize=9)


# ------------------------------
# model loading
# ------------------------------
def build_generator(num_label: int, args, device: torch.device) -> torch.nn.Module:
    netG = Generator(
        args.latent_size,
        num_label,
        d_model=args.G_d_model,
        nhead=args.G_nhead,
        num_layers=args.G_num_layers,
    ).to(device)
    return netG


def load_generator_from_checkpoint(ckpt_path: str, num_label: int, args, device):
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    netG = build_generator(num_label, args, device)

    if isinstance(ckpt, dict) and "netG" in ckpt:
        state_dict = ckpt["netG"]
    else:
        state_dict = ckpt

    missing, unexpected = netG.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    netG.eval()
    return netG


# ------------------------------
# main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_a", type=str, required=True, help="checkpoint A (e.g., with_face_loss)")
    parser.add_argument("--ckpt_b", type=str, required=True, help="checkpoint B (e.g., baseline)")
    parser.add_argument("--tag_a", type=str, default="A", help="tag name for A in output")
    parser.add_argument("--tag_b", type=str, default="B", help="tag name for B in output")

    parser.add_argument("--dataset", type=str, default="crello_mainpart_face")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--start_idx", type=int, default=0, help="deterministic: start sample index in dataset")

    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--G_d_model", type=int, default=256)
    parser.add_argument("--G_nhead", type=int, default=4)
    parser.add_argument("--G_num_layers", type=int, default=8)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="debug_face_compare")

    parser.add_argument("--contain_thresh", type=float, default=0.95)
    parser.add_argument("--iou_green_thresh", type=float, default=0.0, help="IoU>thresh -> green")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # dataset (deterministic selection)
    dataset = get_dataset(args.dataset, args.split, transform=None)
    num_label = dataset.num_classes
    print(f"[dataset] {args.dataset} split={args.split}, num_label={num_label}")
    print(f"[dataset] labels: {getattr(dataset, 'labels', 'N/A')}")

    # deterministic subset via slicing indices
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    indices = list(range(args.start_idx, end_idx))
    subset = torch.utils.data.Subset(dataset, indices)

    loader = GeoDataLoader(subset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))  # deterministic first batch

    boxes = batch.x
    labels = batch.y
    batch_idx = batch.batch

    boxes_dense, mask = to_dense_batch(boxes, batch_idx)       # (B,N,4)
    labels_dense, _ = to_dense_batch(labels, batch_idx)        # (B,N)
    padding_mask = ~mask                                       # True = padding

    B, N, _ = boxes_dense.shape
    print(f"[batch] B={B}, N={N}, seed={args.seed}")

    # fixed z for fairness
    z = torch.randn(B, N, args.latent_size, device=device)

    # load two generators
    netA = load_generator_from_checkpoint(args.ckpt_a, num_label, args, device)
    netB = load_generator_from_checkpoint(args.ckpt_b, num_label, args, device)

    # forward
    labels_dev = labels_dense.to(device)
    pad_dev = padding_mask.to(device)

    with torch.no_grad():
        outA = netA(z, labels_dev, pad_dev)
        outB = netB(z, labels_dev, pad_dev)

    # output dirs
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir_a = os.path.join(args.out_dir, args.tag_a)
    out_dir_b = os.path.join(args.out_dir, args.tag_b)
    os.makedirs(out_dir_a, exist_ok=True)
    os.makedirs(out_dir_b, exist_ok=True)

    # csv summary
    csv_path = os.path.join(args.out_dir, "summary.csv")
    rows: List[Dict[str, object]] = []

    # per-sample save
    for b in range(B):
        valid = mask[b]
        if valid.sum() == 0:
            continue

        lab = labels_dense[b][valid].cpu()
        boxesA = outA[b][valid].cpu()
        boxesB = outB[b][valid].cpu()

        metA = compute_metrics_for_one_layout(
            boxesA, lab, contain_thresh=args.contain_thresh
        )
        metB = compute_metrics_for_one_layout(
            boxesB, lab, contain_thresh=args.contain_thresh
        )

        # save A image
        titleA = (f"{args.tag_a} | faces={int(metA['num_face'])} "
                  f"maxIoU={metA['max_iou']:.3f} "
                  f"meanCov={metA['mean_coverage']:.3f} "
                  f"maxCov={metA['max_coverage']:.3f}")
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_layout(ax, boxesA, lab, titleA, iou_threshold_green=args.iou_green_thresh)
        pathA = os.path.join(out_dir_a, f"sample_{args.start_idx + b:05d}.png")
        fig.savefig(pathA, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # save B image
        titleB = (f"{args.tag_b} | faces={int(metB['num_face'])} "
                  f"maxIoU={metB['max_iou']:.3f} "
                  f"meanCov={metB['mean_coverage']:.3f} "
                  f"maxCov={metB['max_coverage']:.3f}")
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_layout(ax, boxesB, lab, titleB, iou_threshold_green=args.iou_green_thresh)
        pathB = os.path.join(out_dir_b, f"sample_{args.start_idx + b:05d}.png")
        fig.savefig(pathB, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[save] {pathA}")
        print(f"[save] {pathB}")

        row = {
            "global_idx": args.start_idx + b,
            "faces": int(metA["num_face"]),  # same labels for both
            f"{args.tag_a}_maxIoU": metA["max_iou"],
            f"{args.tag_a}_meanCov": metA["mean_coverage"],
            f"{args.tag_a}_maxCov": metA["max_coverage"],
            f"{args.tag_b}_maxIoU": metB["max_iou"],
            f"{args.tag_b}_meanCov": metB["mean_coverage"],
            f"{args.tag_b}_maxCov": metB["max_coverage"],
        }
        rows.append(row)

    # write csv
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[write] {csv_path}")

        # print quick aggregate
        def agg(tag: str):
            max_iou = np.array([r[f"{tag}_maxIoU"] for r in rows], dtype=np.float32)
            mean_cov = np.array([r[f"{tag}_meanCov"] for r in rows], dtype=np.float32)
            max_cov = np.array([r[f"{tag}_maxCov"] for r in rows], dtype=np.float32)
            return max_iou.mean(), mean_cov.mean(), max_cov.mean()

        a_iou, a_mcov, a_xcov = agg(args.tag_a)
        b_iou, b_mcov, b_xcov = agg(args.tag_b)

        print("\n=== Aggregate (over drawn samples) ===")
        print(f"{args.tag_a}: mean(maxIoU)={a_iou:.4f} | mean(meanCov)={a_mcov:.4f} | mean(maxCov)={a_xcov:.4f}")
        print(f"{args.tag_b}: mean(maxIoU)={b_iou:.4f} | mean(meanCov)={b_mcov:.4f} | mean(maxCov)={b_xcov:.4f}")

    else:
        print("[warn] no rows written (maybe empty batch)")

if __name__ == "__main__":
    main()
