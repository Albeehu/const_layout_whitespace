#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#用/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_with_face.pkl
#這個才是有所有label
"""
python make_fixed_sample_v49_n5.py \
  --pkl_path /home/albee/const_layout_whitespace/data/dataset/crello/crello_train_with_face.pkl \
  --num_samples 64 \
  --latent_size 4 \
  --max_nodes 5 \
  --max_faces 4 \
  --pick faceheavy \
  --pool 50000 \
  --min_faces 1 \
  --contain_thr 0.98
  """

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch

# ========= Constants =========
SVG_ID = 0
TEXT_ID = 1
IMG_ID = 2
BG_ID = 3
MASK_ID = 4
FACE_ID = 5


def xywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    """boxes: (N,4) cx,cy,w,h -> (N,4) x1,y1,x2,y2 clipped to [0,1]"""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = np.clip(cx - w / 2.0, 0.0, 1.0)
    y1 = np.clip(cy - h / 2.0, 0.0, 1.0)
    x2 = np.clip(cx + w / 2.0, 0.0, 1.0)
    y2 = np.clip(cy + h / 2.0, 0.0, 1.0)
    return np.stack([x1, y1, x2, y2], axis=1)


def inter_area_np(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """a:(M,4), b:(F,4) -> inter:(M,F)"""
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]), dtype=np.float32)
    x1 = np.maximum(a_xyxy[:, None, 0], b_xyxy[None, :, 0])
    y1 = np.maximum(a_xyxy[:, None, 1], b_xyxy[None, :, 1])
    x2 = np.minimum(a_xyxy[:, None, 2], b_xyxy[None, :, 2])
    y2 = np.minimum(a_xyxy[:, None, 3], b_xyxy[None, :, 3])
    iw = np.clip(x2 - x1, 0.0, None)
    ih = np.clip(y2 - y1, 0.0, None)
    return (iw * ih).astype(np.float32)


def area_np(xyxy: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """xyxy:(N,4)->(N,) area"""
    if xyxy.size == 0:
        return np.zeros((0,), dtype=np.float32)
    w = np.clip(xyxy[:, 2] - xyxy[:, 0], 0.0, None)
    h = np.clip(xyxy[:, 3] - xyxy[:, 1], 0.0, None)
    return np.maximum(w * h, eps).astype(np.float32)


@dataclass
class SamplePack:
    x: torch.Tensor         # (N,) long
    pos: torch.Tensor       # (N,4) float
    mask: torch.Tensor      # (N,) bool
    face_pos: torch.Tensor  # (max_faces,4) float
    face_mask: torch.Tensor # (max_faces,) bool
    face_cnt: int           # number of kept face tokens


class RawLayoutDataset:
    """
    Fixed-sample generation dataset for fixed .pt creation.

    Current keep logic for non-face tokens (especially when max_nodes=5):
    - Face tokens do NOT count toward max_nodes
    - Big SVG (area > 0.15) is relabeled as IMG before truncation
    - Keep policy prefers a clean layout skeleton instead of random priority-only truncation
    - For max_nodes=5, the target skeleton is:
        1 IMG + 2 TEXT + 1 SVG/DECO + 1 EXTRA
      where SVG/DECO means one token from {SVG, BG, MASK}, and EXTRA is filled by
      remaining tokens with priority IMG > TEXT > SVG > BG > MASK.
    - Within each bucket, larger-area elements are preferred.
    - Keep a face if ANY kept IMG contains it with ratio >= contain_thr.
      (best over KEPT images)
    """
    def __init__(
        self,
        data_list: List[Tuple[List[List[float]], List[int]]],
        num_classes: int = 6,
        max_nodes: int = 5,
        max_faces: int = 4,
        contain_thr: float = 0.98,
        base_seed: int = 42,
    ):
        self.data_list = data_list
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.max_faces = max_faces
        self.contain_thr = contain_thr
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> SamplePack:
        rng = np.random.default_rng(self.base_seed + int(idx))

        bbox, label = self.data_list[idx]
        bbox = np.asarray(bbox, dtype=np.float32)
        label = np.asarray(label, dtype=np.int64)

        # split face / others
        face_mask_idx = (label == FACE_ID)
        temp_f_label = label[face_mask_idx]
        temp_f_bbox  = bbox[face_mask_idx]
        o_label = label[~face_mask_idx]
        o_bbox  = bbox[~face_mask_idx]

        # SVG big area -> IMG
        if o_label.size > 0:
            a = o_bbox[:, 2] * o_bbox[:, 3]
            o_label[(o_label == SVG_ID) & (a > 0.15)] = IMG_ID

        # backup pre-trunc lists for consistent indexing
        o_label_all = o_label.copy()
        o_bbox_all = o_bbox.copy()

        n = int(o_label.shape[0])
        keep_o_idxs = np.arange(n, dtype=np.int64)

        # choose non-face tokens with a cleaner skeleton policy
        if n > self.max_nodes:
            idxs = np.arange(n, dtype=np.int64)
            areas = (o_bbox[:, 2] * o_bbox[:, 3]).astype(np.float32)

            def sort_by_area_desc(cand: np.ndarray) -> np.ndarray:
                if cand.size == 0:
                    return cand.astype(np.int64)
                order = np.argsort(-areas[cand], kind="stable")
                return cand[order].astype(np.int64)

            def take_from(cand: np.ndarray, k: int, chosen: List[int]) -> None:
                if k <= 0 or cand.size == 0:
                    return
                chosen_set = set(chosen)
                picked = 0
                sorted_cand = sort_by_area_desc(cand)
                for c in sorted_cand.tolist():
                    if c in chosen_set:
                        continue
                    chosen.append(int(c))
                    chosen_set.add(int(c))
                    picked += 1
                    if len(chosen) >= self.max_nodes or picked >= k:
                        break

            img_idx  = idxs[o_label == IMG_ID]
            text_idx = idxs[o_label == TEXT_ID]
            deco_idx = idxs[(o_label == SVG_ID) | (o_label == BG_ID) | (o_label == MASK_ID)]

            keep_list: List[int] = []

            if self.max_nodes >= 5:
                # target skeleton for clean layouts: 1 IMG + 2 TEXT + 1 DECO + 1 EXTRA
                take_from(img_idx, 1, keep_list)
                take_from(text_idx, 2, keep_list)
                take_from(deco_idx, 1, keep_list)
            else:
                # smaller budgets: still prefer core structure first
                take_from(img_idx, min(1, self.max_nodes), keep_list)
                if len(keep_list) < self.max_nodes:
                    take_from(text_idx, min(2, self.max_nodes - len(keep_list)), keep_list)
                if len(keep_list) < self.max_nodes:
                    take_from(deco_idx, 1, keep_list)

            # fill remaining slots by priority using largest remaining elements
            priority_groups = [
                idxs[o_label == IMG_ID],
                idxs[o_label == TEXT_ID],
                idxs[o_label == SVG_ID],
                idxs[o_label == BG_ID],
                idxs[o_label == MASK_ID],
            ]
            for cand in priority_groups:
                if len(keep_list) >= self.max_nodes:
                    break
                take_from(cand, self.max_nodes - len(keep_list), keep_list)

            keep = np.array(sorted(keep_list), dtype=np.int64)
            keep_o_idxs = keep.copy()
            o_label = o_label[keep]
            o_bbox  = o_bbox[keep]

        # ---- Face filtering: keep face if ANY kept image contains it ----
        f_label = temp_f_label
        f_bbox  = temp_f_bbox

        if f_label.size > 0:
            # identify kept IMG indices in the "all others" index space
            kept_set = set(keep_o_idxs.tolist())
            img_all_idx = np.where(o_label_all == IMG_ID)[0].astype(np.int64)
            kept_img_global_idx = np.array([i for i in img_all_idx if int(i) in kept_set], dtype=np.int64)

            if kept_img_global_idx.size == 0:
                f_keep = np.zeros((f_label.shape[0],), dtype=bool)
            else:
                img_xyxy = xywh_to_xyxy_np(o_bbox_all[kept_img_global_idx])
                face_xyxy = xywh_to_xyxy_np(f_bbox)

                inter = inter_area_np(img_xyxy, face_xyxy)      # (M_kept, F)
                f_area = area_np(face_xyxy)                     # (F,)
                ratio = inter / f_area[None, :]                 # (M_kept, F)

                best_ratio = ratio.max(axis=0)                  # best over KEPT images
                f_keep = (best_ratio >= float(self.contain_thr))

            f_label = f_label[f_keep]
            f_bbox  = f_bbox[f_keep]

        # clip to max_faces
        if f_label.size > 0:
            f_label = f_label[: self.max_faces]
            f_bbox  = f_bbox[: self.max_faces]

        # face_pos/face_mask (for anchor)
        face_pos = torch.zeros((self.max_faces, 4), dtype=torch.float32)
        face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
        nf = int(min(f_label.shape[0], self.max_faces))
        if nf > 0:
            face_pos[:nf] = torch.from_numpy(f_bbox[:nf]).float()
            face_mask[:nf] = True

        # merge + pad
        final_label = np.concatenate([o_label, f_label], axis=0) if f_label.size > 0 else o_label
        final_bbox  = np.concatenate([o_bbox,  f_bbox],  axis=0) if f_label.size > 0 else o_bbox

        total_cap = self.max_nodes + self.max_faces
        pad_x = torch.full((total_cap,), self.num_classes - 1, dtype=torch.long)
        pad_pos = torch.zeros((total_cap, 4), dtype=torch.float32)
        pad_mask = torch.zeros((total_cap,), dtype=torch.bool)

        curr_total = int(final_label.shape[0])
        if curr_total > 0:
            pad_x[:curr_total] = torch.from_numpy(final_label.astype(np.int64)).long()
            pad_pos[:curr_total] = torch.from_numpy(final_bbox.astype(np.float32)).float()
            pad_mask[:curr_total] = True

        face_cnt = int(((pad_x == FACE_ID) & pad_mask).sum().item())

        return SamplePack(
            x=pad_x, pos=pad_pos, mask=pad_mask,
            face_pos=face_pos, face_mask=face_mask,
            face_cnt=face_cnt
        )


def build_fixed_sample(
    ds: RawLayoutDataset,
    num_samples: int,
    latent_size: int,
    pick: str,
    pool: int,
    min_faces: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_total = len(ds)

    if pick == "first":
        cand_idxs = list(range(min(pool, n_total)))
    else:
        pool_eff = min(pool, n_total)
        cand_idxs = rng.choice(n_total, size=pool_eff, replace=False).tolist()

    scored: List[Tuple[int, int]] = []
    all_items: Dict[int, SamplePack] = {}

    for i in cand_idxs:
        item = ds[int(i)]
        all_items[int(i)] = item
        scored.append((item.face_cnt, int(i)))

    # sort by face count descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # pick strategy
    picked: List[int] = []

    if pick == "faceheavy":
        # take those >=min_faces first
        good = [i for (fc, i) in scored if fc >= min_faces]
        if len(good) >= num_samples:
            picked = good[:num_samples]
        else:
            # not enough — degrade gracefully, fill with best remaining
            print(f"[WARN] only {len(good)} samples have face_cnt >= {min_faces}; "
                  f"will fill the rest with the best remaining candidates.")
            picked = good
            remain = [i for (_, i) in scored if i not in set(picked)]
            need = num_samples - len(picked)
            picked.extend(remain[:need])
    elif pick == "random":
        picked = cand_idxs[:min(num_samples, len(cand_idxs))]
        if len(picked) < num_samples:
            # fallback sample more
            rest = [i for i in range(n_total) if i not in set(picked)]
            extra = rng.choice(rest, size=num_samples - len(picked), replace=False).tolist()
            picked.extend(extra)
    else:  # "first"
        picked = cand_idxs[:min(num_samples, len(cand_idxs))]
        if len(picked) < num_samples:
            rest = [i for i in range(n_total) if i not in set(picked)]
            picked.extend(rest[: (num_samples - len(picked))])

    # materialize batch
    batch = [all_items[i] if i in all_items else ds[i] for i in picked]
    label = torch.stack([b.x for b in batch], dim=0)          # (B,N)
    pos   = torch.stack([b.pos for b in batch], dim=0)        # (B,N,4)
    mask  = torch.stack([b.mask for b in batch], dim=0)       # (B,N)
    face_pos  = torch.stack([b.face_pos for b in batch], dim=0)
    face_mask = torch.stack([b.face_mask for b in batch], dim=0)

    B, N = label.shape
    torch.manual_seed(seed)
    z = torch.randn((B, N, latent_size), dtype=torch.float32)

    # stats
    face_counts = [int(((label[b] == FACE_ID) & mask[b]).sum().item()) for b in range(B)]

    return {
        "label": label,
        "pos": pos,
        "mask": mask,
        "z": z,
        "face_pos": face_pos,
        "face_mask": face_mask,
        "meta": {
            "seed": seed,
            "pick": pick,
            "picked_indices": picked,
            "pool": pool,
            "min_faces": min_faces,
            "latent_size": latent_size,
            "max_nodes": ds.max_nodes,
            "max_faces": ds.max_faces,
            "contain_thr": ds.contain_thr,
            "base_seed": ds.base_seed,
            "faces_per_sample": face_counts,
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_path", type=str, required=True)
    ap.add_argument("--out", type=str, default="fixed_sample_v49_n5.pt")
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--latent_size", type=int, default=4)
    ap.add_argument("--max_nodes", type=int, default=5)
    ap.add_argument("--max_faces", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_seed", type=int, default=12345, help="deterministic tie-break per idx")
    ap.add_argument("--pick", type=str, default="faceheavy", choices=["faceheavy", "random", "first"])
    ap.add_argument("--pool", type=int, default=50000)
    ap.add_argument("--min_faces", type=int, default=1)
    ap.add_argument("--contain_thr", type=float, default=0.98)
    args = ap.parse_args()

    # seeds for overall process
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(args.pkl_path)

    with open(args.pkl_path, "rb") as f:
        data_list = pickle.load(f)

    ds = RawLayoutDataset(
        data_list=data_list,
        num_classes=args.num_classes,
        max_nodes=args.max_nodes,
        max_faces=args.max_faces,
        contain_thr=args.contain_thr,
        base_seed=args.base_seed,
    )

    ck = build_fixed_sample(
        ds=ds,
        num_samples=args.num_samples,
        latent_size=args.latent_size,
        pick=args.pick,
        pool=args.pool,
        min_faces=args.min_faces,
        seed=args.seed,
    )

    torch.save(ck, args.out)
    print(f"[OK] saved: {args.out}")

    # summary
    tmp = ck["label"][ck["mask"]].reshape(-1)
    binc = torch.bincount(tmp, minlength=args.num_classes).tolist()
    faces = ck["meta"]["faces_per_sample"]
    print("[fixed bincount]", binc)
    print(f"[faces] mean={float(np.mean(faces)):.3f} max={int(np.max(faces))} min={int(np.min(faces))}")
    print("[faces top10]", sorted(faces, reverse=True)[:10])


if __name__ == "__main__":
    main()