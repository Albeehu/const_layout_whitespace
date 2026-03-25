"""用 train_fixed_v6_0.6pkl_v53 改 v54
1. 新增 content_region_loss 直接看「所有前景元素的 union box」
讓整體內容區落在某個 template 指定區域
這樣留白會比較像被設計出來，不是元素亂縮後剩下的空白
順便加 content_union_stats，讓 log 裡直接看到內容區寬高與左上角位置
"""
import os
import csv
import argparse
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import random
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from metric import LayoutFID, compute_maximum_iou
from model.layoutganpp import Generator, Discriminator
from data.util import LexicographicSort, HorizontalFlip
from util import init_experiment, save_image, save_checkpoint

# ========= Constants (保留你原本 v10/v12 的命名) =========
SVG_ID = 0
TEXT_ID = 1  
IMG_ID = 2   # 圖片元素容器
BG_ID = 3    # 背景
MASK_ID = 4  # 遮罩
FACE_ID = 5

# ===== 固定使用的訓練 PKL 路徑 =====
PKL_PATH_CRELLO_FULL = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_all.pkl"
PKL_PATH_HIGH_QUALITY = "/home/albee/const_layout_whitespace/data/dataset/crello/high_quality.pkl"
PKL_PATH_WS_60 = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_ws_gt0.6_with_face.pkl"

def seed_worker(worker_id: int):
    """Ensure numpy/random have different seeds across DataLoader workers.

    Keep numpy/random seeds different across workers for augmentation randomness.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_training_checkpoint(ckpt_dir, iteration, netG, netD, optimizerG, optimizerD, args):
    """Save a resumable checkpoint for training."""
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        'iteration': int(iteration),
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'args': vars(args),
    }
    latest_path = os.path.join(ckpt_dir, 'ckpt_latest.pth')
    numbered_path = os.path.join(ckpt_dir, f'ckpt_{int(iteration):05d}.pth')
    torch.save(state, latest_path)
    torch.save(state, numbered_path)
    print(f"[Checkpoint] saved: {numbered_path}")
    print(f"[Checkpoint] updated latest: {latest_path}")


def load_training_checkpoint(ckpt_path, device, netG, netD, optimizerG, optimizerD):
    """Load model / optimizer states and return the restored iteration."""
    if not ckpt_path:
        return 0
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'resume checkpoint not found: {ckpt_path}')

    print(f"[Resume] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt['netG'])
    netD.load_state_dict(ckpt['netD'])
    optimizerG.load_state_dict(ckpt['optimizerG'])
    optimizerD.load_state_dict(ckpt['optimizerD'])
    restored_iteration = int(ckpt.get('iteration', 0))
    print(f"[Resume] restored iteration = {restored_iteration}")
    return restored_iteration



def select_nonface_indices_by_role(o_bbox, o_label, max_nodes):
    """Select non-face tokens with a layout-friendly policy.

    Priority for max_nodes=5 goal:
      1) reserve 1 image
      2) reserve 2 text
      3) reserve 1 svg/deco
      4) fill the remaining slots by priority (Image > Text > SVG > BG > Mask)

    Within each category, larger-area elements are kept first instead of random sampling.
    This is more stable for high-whitespace / clean-layout training than pure random truncation.
    """
    n = len(o_label)
    if n <= max_nodes:
        return np.arange(n, dtype=np.int64)

    idxs = np.arange(n, dtype=np.int64)
    area = o_bbox[:, 2] * o_bbox[:, 3]
    used = np.zeros(n, dtype=bool)
    keep_list = []

    def _take(label_id, k):
        nonlocal keep_list
        if k <= 0:
            return
        cand = idxs[(o_label == label_id) & (~used)]
        if cand.size == 0:
            return
        order = np.argsort(-area[cand], kind='stable')
        chosen = cand[order[:min(k, cand.size)]]
        used[chosen] = True
        keep_list.extend(chosen.tolist())

    if max_nodes >= 4:
        _take(IMG_ID, 1)
        _take(TEXT_ID, min(2, max_nodes - len(keep_list)))
        _take(SVG_ID, min(1, max_nodes - len(keep_list)))
    elif max_nodes == 3:
        _take(IMG_ID, 1)
        _take(TEXT_ID, min(1, max_nodes - len(keep_list)))
        _take(SVG_ID, min(1, max_nodes - len(keep_list)))
    elif max_nodes == 2:
        _take(IMG_ID, 1)
        _take(TEXT_ID, min(1, max_nodes - len(keep_list)))
    elif max_nodes == 1:
        _take(IMG_ID, 1)

    fill_priority = (IMG_ID, TEXT_ID, SVG_ID, BG_ID, MASK_ID)
    for label_id in fill_priority:
        need = max_nodes - len(keep_list)
        if need <= 0:
            break
        _take(label_id, need)

    if len(keep_list) < max_nodes:
        cand = idxs[~used]
        if cand.size > 0:
            order = np.argsort(-area[cand], kind='stable')
            chosen = cand[order[:min(max_nodes - len(keep_list), cand.size)]]
            keep_list.extend(chosen.tolist())

    keep = np.array(sorted(keep_list[:max_nodes]), dtype=np.int64)
    return keep

class RawLayoutDataset(torch.utils.data.Dataset):
    """
    更新邏輯（Robust + max_nodes=5 友善版）：
    1) Face (ID=5) 不計入 max_nodes 的配額。
    2) 其他元件優先保留 1 Image + 2 Text + 1 SVG/Deco，剩餘名額再依 priority 補齊。
    3) Face 只保留「屬於被保留 Image」的人臉（避免 face 對應到被截斷掉的 image）。
    4) 回傳 face 的 target 以「相對於對應 Image 的座標」表示：face_rel = (rx, ry, rw, rh)
       - rx = (cx_face - cx_img) / w_img
       - ry = (cy_face - cy_img) / h_img
       - rw = w_face / w_img
       - rh = h_face / h_img
    5) Robust: 如果該樣本沒有 image（或 image 全被截斷），則直接丟棄所有 face，避免 rel=空陣列造成 tensor assignment error。
    """
    def __init__(
        self,
        data_list,
        num_classes,
        max_nodes=5,
        max_faces=4,
        colors=None,
        aug_hflip: bool = False,
        aug_vflip: bool = False,
        flip_prob: float = 0.5,
    ):
        self.data_list = data_list
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.max_faces = max_faces
        self.colors = colors
        self.aug_hflip = bool(aug_hflip)
        self.aug_vflip = bool(aug_vflip)
        self.flip_prob = float(flip_prob)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        bbox, label = self.data_list[idx]
        bbox = np.array(bbox)
        label = np.array(label)

        # ----------------------------
        # Data augmentation: horizontal / vertical flip (cx,cy,w,h in [0,1])
        # NOTE: do this BEFORE face->image mapping and face_rel computation, so
        #       face2img mapping stays valid and face_rel target stays consistent.
        # ----------------------------
        if bbox.size != 0 and (self.aug_hflip or self.aug_vflip) and self.flip_prob > 0:
            if self.aug_hflip and (np.random.rand() < self.flip_prob):
                bbox[:, 0] = 1.0 - bbox[:, 0]
            if self.aug_vflip and (np.random.rand() < self.flip_prob):
                bbox[:, 1] = 1.0 - bbox[:, 1]

        # 1) 分離人臉與其他元件
        face_mask_idx = (label == FACE_ID)
        temp_f_label = label[face_mask_idx]
        temp_f_bbox = bbox[face_mask_idx]
        o_label = label[~face_mask_idx]
        o_bbox = bbox[~face_mask_idx]

        # 2) 處理非人臉元件 (把大 SVG 當 Image)
        if len(o_label) > 0:
            area = o_bbox[:, 2] * o_bbox[:, 3]
            o_label[(o_label == SVG_ID) & (area > 0.15)] = IMG_ID

        # 備份（在 max_nodes 截斷前）用於 face->image 對應
        o_label_all = o_label.copy()
        o_bbox_all = o_bbox.copy()

        # 2.5) 如果本來就沒有任何非-face token，無法建立 image 對應 => 直接丟棄 face
        if len(o_label_all) == 0:
            temp_f_label = np.array([], dtype=np.int64)
            temp_f_bbox = np.empty((0, 4), dtype=np.float32)

        n = len(o_label)
        keep_o_idxs = np.arange(n, dtype=np.int64)  # o_label_all/o_bbox_all 中被保留的索引（截斷前的 index）
        if n > self.max_nodes:
            keep = select_nonface_indices_by_role(o_bbox, o_label, self.max_nodes)
            keep_o_idxs = keep.copy()
            o_label = o_label[keep]
            o_bbox = o_bbox[keep]

        # 3) Face 只保留「屬於被保留 Image」的人臉，並建立 face->image mapping
        f_label = temp_f_label
        f_bbox = temp_f_bbox

        keep_map = {int(orig_i): int(pos_i) for pos_i, orig_i in enumerate(keep_o_idxs.tolist())}

        face2img_list = []
        if len(f_label) > 0 and len(o_label_all) > 0:
            img_all_idx = np.where(o_label_all == IMG_ID)[0]

            if img_all_idx.size == 0:
                # 沒有 image => 無法對應 => 丟棄全部 face
                f_label = np.array([], dtype=np.int64)
                f_bbox = np.empty((0, 4), dtype=np.float32)
                face2img_list = []
            else:
                def _cxcywh_to_xyxy_np(bb):
                    cx, cy, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
                    x1 = np.clip(cx - w / 2.0, 0.0, 1.0)
                    y1 = np.clip(cy - h / 2.0, 0.0, 1.0)
                    x2 = np.clip(cx + w / 2.0, 0.0, 1.0)
                    y2 = np.clip(cy + h / 2.0, 0.0, 1.0)
                    return np.stack([x1, y1, x2, y2], axis=1)

                img_xy = _cxcywh_to_xyxy_np(o_bbox_all[img_all_idx])  # (M,4)
                face_xy = _cxcywh_to_xyxy_np(f_bbox)                  # (F,4)

                x1 = np.maximum(img_xy[:, None, 0], face_xy[None, :, 0])
                y1 = np.maximum(img_xy[:, None, 1], face_xy[None, :, 1])
                x2 = np.minimum(img_xy[:, None, 2], face_xy[None, :, 2])
                y2 = np.minimum(img_xy[:, None, 3], face_xy[None, :, 3])
                inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

                f_area = np.clip(face_xy[:, 2] - face_xy[:, 0], 0.0, None) * np.clip(face_xy[:, 3] - face_xy[:, 1], 0.0, None)
                f_area = np.maximum(f_area, 1e-9)

                ratio = inter / f_area[None, :]  # (M,F)
                best_m = ratio.argmax(axis=0)    # (F,)
                best_ratio = ratio[best_m, np.arange(len(f_label))]

                contain_thr = 0.98
                assigned_img_idx = np.where(best_ratio >= contain_thr, img_all_idx[best_m], -1).astype(np.int64)

                f_keep = np.zeros((len(f_label),), dtype=bool)
                face2img_tmp = np.full((len(f_label),), -1, dtype=np.int64)

                for fi, ai in enumerate(assigned_img_idx.tolist()):
                    if ai == -1:
                        continue
                    if ai in keep_map:
                        f_keep[fi] = True
                        face2img_tmp[fi] = keep_map[ai]

                f_label = f_label[f_keep]
                f_bbox = f_bbox[f_keep]
                face2img_tmp = face2img_tmp[f_keep]
                face2img_list = face2img_tmp.tolist()

        # 若沒有臉，統一成空
        if len(f_label) == 0:
            f_label = np.array([], dtype=np.int64)
            f_bbox = np.empty((0, 4), dtype=np.float32)
            face2img_list = []

        # 4) 建立「相對座標」target：face_rel_pos，以及 face2img / face_mask
        face_rel_pos = torch.zeros((self.max_faces, 4), dtype=torch.float)
        face2img = torch.full((self.max_faces,), -1, dtype=torch.long)
        face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)

        # Robust: nf 以 mapping 長度為準，避免 face2img_list 不足造成 rel=空
        nf = min(len(face2img_list), len(f_label), self.max_faces)
        if nf > 0:
            eps = 1e-6
            f_bbox_nf = f_bbox[:nf].astype(np.float32)
            f2i_nf = np.array(face2img_list[:nf], dtype=np.int64)

            # 用截斷後的 image bbox (o_bbox) 來算相對座標
            img_bbox_nf = o_bbox[f2i_nf].astype(np.float32)

            rx = (f_bbox_nf[:, 0] - img_bbox_nf[:, 0]) / (img_bbox_nf[:, 2] + eps)
            ry = (f_bbox_nf[:, 1] - img_bbox_nf[:, 1]) / (img_bbox_nf[:, 3] + eps)
            rw = (f_bbox_nf[:, 2]) / (img_bbox_nf[:, 2] + eps)
            rh = (f_bbox_nf[:, 3]) / (img_bbox_nf[:, 3] + eps)

            rel = np.stack([rx, ry, rw, rh], axis=1).astype(np.float32)

            face_rel_pos[:nf] = torch.from_numpy(rel)
            face2img[:nf] = torch.from_numpy(f2i_nf)
            face_mask[:nf] = True

        # 5) 合併與 Padding：token 序列仍然是 [o_label + f_label]
        final_label = np.concatenate([o_label, f_label])
        final_bbox = np.concatenate([o_bbox, f_bbox]) if len(f_bbox) > 0 else o_bbox

        total_cap = self.max_nodes + self.max_faces
        pad_x = torch.full((total_cap,), self.num_classes - 1, dtype=torch.long)
        pad_pos = torch.zeros((total_cap, 4), dtype=torch.float)
        pad_mask = torch.zeros((total_cap,), dtype=torch.bool)

        curr_total = len(final_label)
        pad_x[:curr_total] = torch.LongTensor(final_label)
        pad_pos[:curr_total] = torch.FloatTensor(final_bbox)
        pad_mask[:curr_total] = True

        return {
            'x': pad_x,
            'pos': pad_pos,
            'mask': pad_mask,
            'face_rel': face_rel_pos,
            'face2img': face2img,
            'face_mask': face_mask,
        }


class WeightedMixedRawDataset(torch.utils.data.Dataset):
    """
    將多個 pkl dataset 混合成單一訓練集，並用 source_weights 控制被抽到的機率。

    使用方式：
      - datasets: list[RawLayoutDataset]
      - source_weights: 與 datasets 等長，例如 [0.3, 0.3, 0.4]
      - epoch_size: 每個 epoch 視為有多少筆 sample

    注意：
      1) __len__ 不再等於真實資料總數，而是你想要每個 epoch 抽幾筆。
      2) 每次 __getitem__ 都會先依權重抽 dataset，再從該 dataset 隨機抽一筆。
      3) 這樣可以避免完整 crello.pkl 數量太大，直接把 high_quality / 60%_ws 淹沒掉。
    """
    def __init__(self, datasets, source_weights, epoch_size=50000, seed=None):
        assert len(datasets) > 0, "datasets 不能是空的"
        assert len(datasets) == len(source_weights), "datasets 跟 source_weights 長度必須一致"

        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.epoch_size = int(epoch_size)
        self.seed = seed

        w = np.asarray(source_weights, dtype=np.float64)
        if np.any(w < 0):
            raise ValueError(f"source_weights 不能有負數: {source_weights}")
        if np.allclose(w.sum(), 0.0):
            raise ValueError("source_weights 總和不能是 0")
        self.source_weights = (w / w.sum()).astype(np.float64)

        # 直接沿用第一個 dataset 的 metadata 給外部使用
        self.num_classes = datasets[0].num_classes
        self.colors = getattr(datasets[0], 'colors', None)

        # 若指定 seed，建立自己的 RNG；否則走全域 np.random
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def __len__(self):
        return self.epoch_size

    def _randint(self, low, high):
        if self.rng is not None:
            return int(self.rng.integers(low, high))
        return int(np.random.randint(low, high))

    def _choice_dataset(self):
        if self.rng is not None:
            return int(self.rng.choice(len(self.datasets), p=self.source_weights))
        return int(np.random.choice(len(self.datasets), p=self.source_weights))

    def __getitem__(self, idx):
        ds_idx = self._choice_dataset()
        sample_idx = self._randint(0, self.lengths[ds_idx])
        return self.datasets[ds_idx][sample_idx]



def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2.0).clamp(0.0, 1.0)
    y1 = (cy - h / 2.0).clamp(0.0, 1.0)
    x2 = (cx + w / 2.0).clamp(0.0, 1.0)
    y2 = (cy + h / 2.0).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_intersection_area_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = (x2 - x1).clamp(min=0); inter_h = (y2 - y1).clamp(min=0)
    return inter_w * inter_h

def box_area_xyxy(xyxy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0); h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
    return (w * h).clamp(min=eps)


def project_fixed_aspect_scale(
    bbox_pred: torch.Tensor,
    bbox_ref: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    target_ids=(SVG_ID, TEXT_ID, IMG_ID),
    eps: float = 1e-6,
    s_min: float = 0.25,
    s_max: float = 4.0,
) -> torch.Tensor:
    """Fix aspect ratio to match bbox_ref, but allow uniform scaling to keep bbox_pred area.

    bbox_pred, bbox_ref: (B, N, 4) in (cx, cy, w, h), normalized to [0,1]
    padding_mask: (B, N) where True means padding (invalid)
    Returns a new tensor (non-inplace) safe for autograd.
    """
    device = bbox_pred.device
    valid = ~padding_mask

    tgt = torch.zeros_like(valid, dtype=torch.bool, device=device)
    for tid in target_ids:
        tgt = tgt | ((label == tid) & valid)

    if not tgt.any():
        return bbox_pred

    w0 = bbox_ref[..., 2].clamp_min(eps)
    h0 = bbox_ref[..., 3].clamp_min(eps)
    w  = bbox_pred[..., 2].clamp_min(eps)
    h  = bbox_pred[..., 3].clamp_min(eps)

    s = torch.sqrt((w * h) / (w0 * h0)).clamp(s_min, s_max)
    w_new = (w0 * s).clamp(min=eps, max=1.0)
    h_new = (h0 * s).clamp(min=eps, max=1.0)

    cx = bbox_pred[..., 0]
    cy = bbox_pred[..., 1]
    replaced = torch.stack([cx, cy, w_new, h_new], dim=-1)
    out = torch.where(tgt.unsqueeze(-1), replaced, bbox_pred)

    half_w = out[..., 2] / 2.0
    half_h = out[..., 3] / 2.0
    cx2 = out[..., 0].clamp(half_w, 1.0 - half_w)
    cy2 = out[..., 1].clamp(half_h, 1.0 - half_h)
    return torch.stack([cx2, cy2, out[..., 2], out[..., 3]], dim=-1)

# ============================================================
# HARD COUPLING: Face is NOT an independent layout element.
# Face token outputs RELATIVE parameters (rx, ry, rw, rh) w.r.t.
# its corresponding Image token. Absolute face bbox is derived:
#   cx_f = cx_img + rx * w_img
#   cy_f = cy_img + ry * h_img
#   w_f  = rw * w_img
#   h_f  = rh * h_img
# ============================================================

def _decode_face_rel_from_token(
    t: torch.Tensor,
    rx_max: float = 0.75,
    ry_max: float = 0.75,
    rw_min: float = 0.05,
    rw_max: float = 1.00,
    rh_min: float = 0.05,
    rh_max: float = 1.00,
) -> torch.Tensor:
    """
    t: (...,4) in [0,1] (generator output)
    returns rel: (...,4) where
      rx, ry in [-rx_max, rx_max] / [-ry_max, ry_max]
      rw in [rw_min, rw_max]
      rh in [rh_min, rh_max]
    """
    tx, ty, tw, th = t.unbind(-1)
    rx = (tx - 0.5) * 2.0 * rx_max
    ry = (ty - 0.5) * 2.0 * ry_max
    rw = rw_min + tw * (rw_max - rw_min)
    rh = rh_min + th * (rh_max - rh_min)
    return torch.stack([rx, ry, rw, rh], dim=-1)

def hard_couple_faces_to_images(
    bbox_pred: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    face2img: torch.Tensor,
    face_mask: Optional[torch.Tensor] = None,
    face_rel_gt: Optional[torch.Tensor] = None,
    use_gt_face_rel: bool = False,
    keep_inside_image: bool = True,
    rx_max: float = 0.75,
    ry_max: float = 0.75,
    rw_min: float = 0.05,
    rw_max: float = 1.00,
    rh_min: float = 0.05,
    rh_max: float = 1.00,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    bbox_pred: (B,N,4) generator output in [0,1]
      - Image/Text/SVG tokens are absolute (cx,cy,w,h)
      - Face tokens are NOT treated as independent final boxes.

    face2img:   (B,max_faces) mapping for i-th face (in order) -> image token index
    face_mask:  (B,max_faces) valid GT faces
    face_rel_gt:(B,max_faces,4) GT relative face box [rx, ry, rw, rh] w.r.t. mapped image box

    If use_gt_face_rel=True and face_rel_gt is provided, face absolute boxes are reconstructed
    from GT relative coordinates + CURRENT image prediction. This means faces never "guess"
    their own positions and always stay coupled to the mapped image.
    """
    B, N, _ = bbox_pred.shape
    bbox_out = bbox_pred.clone()
    max_faces = face2img.size(1)

    for b in range(B):
        valid = ~padding_mask[b]
        f_idx = torch.where((label[b] == FACE_ID) & valid)[0]

        if face_mask is not None:
            n_gt = int(face_mask[b].sum().item())
            n = min(int(f_idx.numel()), n_gt, max_faces)
        else:
            n = min(int(f_idx.numel()), max_faces)

        if n <= 0:
            continue

        for i in range(n):
            img_idx = int(face2img[b, i].item())
            if img_idx < 0 or img_idx >= N:
                continue
            if (not valid[img_idx].item()) or (label[b, img_idx].item() != IMG_ID):
                continue

            if use_gt_face_rel and face_rel_gt is not None:
                rel = face_rel_gt[b, i]
            else:
                rel = _decode_face_rel_from_token(
                    bbox_pred[b, f_idx[i]],
                    rx_max=rx_max, ry_max=ry_max,
                    rw_min=rw_min, rw_max=rw_max,
                    rh_min=rh_min, rh_max=rh_max,
                )
            rx, ry, rw, rh = rel.unbind(-1)

            img_box = bbox_pred[b, img_idx].clone()
            cx_i, cy_i, w_i, h_i = img_box.unbind(-1)
            w_i = w_i.clamp_min(eps)
            h_i = h_i.clamp_min(eps)

            # derive absolute face bbox from current image + relative face params
            # torch.clamp does not accept keyword min/max when one bound is a Tensor
            # on some PyTorch versions, so use maximum/minimum explicitly.
            w_f = torch.minimum(torch.maximum(rw * w_i, torch.full_like(w_i, eps)), w_i)
            h_f = torch.minimum(torch.maximum(rh * h_i, torch.full_like(h_i, eps)), h_i)
            cx_f = cx_i + rx * w_i
            cy_f = cy_i + ry * h_i

            if keep_inside_image:
                left_i   = cx_i - w_i / 2.0
                right_i  = cx_i + w_i / 2.0
                top_i    = cy_i - h_i / 2.0
                bottom_i = cy_i + h_i / 2.0
                cx_f = cx_f.clamp(left_i + w_f / 2.0, right_i - w_f / 2.0)
                cy_f = cy_f.clamp(top_i + h_f / 2.0, bottom_i - h_f / 2.0)

            # also keep inside canvas for safety
            cx_f = cx_f.clamp(w_f / 2.0, 1.0 - w_f / 2.0)
            cy_f = cy_f.clamp(h_f / 2.0, 1.0 - h_f / 2.0)

            bbox_out[b, f_idx[i]] = torch.stack([cx_f, cy_f, w_f, h_f], dim=-1)

    return bbox_out


def build_face_rel_gt_from_reference(
    bbox_ref: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    face2img: torch.Tensor,
    max_faces: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build GT relative face boxes [rx, ry, rw, rh] from absolute reference bboxes
    and an explicit face->image mapping. The order follows face token order.
    """
    B, N, _ = bbox_ref.shape
    face_rel = torch.zeros((B, max_faces, 4), device=bbox_ref.device, dtype=bbox_ref.dtype)

    for b in range(B):
        valid = ~padding_mask[b]
        f_idx = torch.where((label[b] == FACE_ID) & valid)[0]
        nf = min(int(f_idx.numel()), max_faces)
        for i in range(nf):
            img_idx = int(face2img[b, i].item())
            if img_idx < 0 or img_idx >= N:
                continue
            if (not valid[img_idx].item()) or (label[b, img_idx].item() != IMG_ID):
                continue

            face_box = bbox_ref[b, f_idx[i]]
            img_box = bbox_ref[b, img_idx]
            cx_f, cy_f, w_f, h_f = face_box.unbind(-1)
            cx_i, cy_i, w_i, h_i = img_box.unbind(-1)
            w_i = w_i.clamp_min(eps)
            h_i = h_i.clamp_min(eps)

            rx = (cx_f - cx_i) / w_i
            ry = (cy_f - cy_i) / h_i
            rw = (w_f / w_i).clamp(min=0.0, max=1.0)
            rh = (h_f / h_i).clamp(min=0.0, max=1.0)
            face_rel[b, i] = torch.stack([rx, ry, rw, rh], dim=-1)

    return face_rel


def infer_face2img_from_reference(
    bbox_ref: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    max_faces: int = 4,
    contain_thr: float = 0.98,
) -> torch.Tensor:
    """
    用 reference bbox (通常是 GT 的 pos) 推導 face -> image 的 mapping。
    回傳 face2img: (B, max_faces)；以 face token 的順序為準（取前 max_faces）。
    """
    device = bbox_ref.device
    B, N, _ = bbox_ref.shape
    face2img = torch.full((B, max_faces), -1, device=device, dtype=torch.long)

    for b in range(B):
        valid = ~padding_mask[b]
        img_idx = torch.where((label[b] == IMG_ID) & valid)[0]
        face_idx = torch.where((label[b] == FACE_ID) & valid)[0]
        if img_idx.numel() == 0 or face_idx.numel() == 0:
            continue

        out_xyxy = xywh_to_xyxy(bbox_ref[b][img_idx])   # (M,4)
        in_xyxy  = xywh_to_xyxy(bbox_ref[b][face_idx])  # (F,4)

        in_area = box_area_xyxy(in_xyxy).clamp_min(1e-9)
        inter_area = box_intersection_area_xyxy(out_xyxy, in_xyxy)  # (M,F)
        contain_ratio = (inter_area / in_area).max(dim=0)[0]        # (F,)
        best_m = (inter_area / in_area).argmax(dim=0)               # (F,)

        nf = min(int(face_idx.numel()), max_faces)
        for i in range(nf):
            if contain_ratio[i].item() >= contain_thr:
                face2img[b, i] = img_idx[best_m[i]].long()

    return face2img



def get_curr_lambda(base_lambda: float, curr_iter: int, start_iter: int = 0, warmup_iter: int = 0) -> float:
    """Linear warmup helper."""
    if curr_iter < start_iter:
        return 0.0
    if warmup_iter <= 0:
        return float(base_lambda)
    alpha = min(1.0, max(0.0, float(curr_iter - start_iter) / float(max(1, warmup_iter))))
    return float(base_lambda) * alpha


def get_stage_loss_multipliers(stage: str):
    """Training curriculum switches.

    baseline   : only adversarial + overlap + basic geometry safety
    text       : baseline + text-related losses
    text_image : text + image geometry / prior
    full       : all losses
    """
    if stage == 'baseline':
        return {
            'style': 0.0, 'template': 0.0, 'space': 0.0, 'align': 0.0,
            'text_width': 0.0, 'text_geom': 0.0, 'img_geom': 0.0,
            'shape_prior': 0.0, 'role_layout': 0.0, 'region': 0.0,
            'face': 0.0, 'cont': 0.0, 'anchor': 0.0,
        }
    if stage == 'text':
        return {
            'style': 0.0, 'template': 0.0, 'space': 0.0, 'align': 0.0,
            'text_width': 1.0, 'text_geom': 1.0, 'img_geom': 0.0,
            'shape_prior': 0.0, 'role_layout': 0.0, 'region': 0.0,
            'face': 0.0, 'cont': 0.0, 'anchor': 0.0,
        }
    if stage == 'text_image':
        return {
            'style': 0.0, 'template': 0.0, 'space': 0.0, 'align': 0.0,
            'text_width': 1.0, 'text_geom': 1.0, 'img_geom': 1.0,
            'shape_prior': 0.5, 'role_layout': 0.0, 'region': 0.0,
            'face': 0.0, 'cont': 0.0, 'anchor': 0.0,
        }
    if stage == 'full':
        return {
            'style': 1.0, 'template': 1.0, 'space': 1.0, 'align': 1.0,
            'text_width': 1.0, 'text_geom': 1.0, 'img_geom': 1.0,
            'shape_prior': 1.0, 'role_layout': 1.0, 'region': 1.0,
            'face': 1.0, 'cont': 1.0, 'anchor': 1.0,
        }
    raise ValueError(f'unknown train_stage: {stage}')


def make_style_tags(args, batch_size: int, device: torch.device):
    """Build style tags for whitespace_style_loss.

    template mode: keep template-conditioned style target
    random mode  : genuinely randomize style target across samples
    none mode    : pass None so whitespace_style_loss samples internally
    """
    if args.ws_style_mode != 'random':
        return None

    if args.style_conditioning == 'template':
        style_map = {
            'top_hero': 2,
            'center_vertical': 0,
            'right_column': 1,
            'bottom_caption': 2,
            'hybrid': 3,
        }
        return torch.full((batch_size,), style_map[args.layout_template], device=device, dtype=torch.long)

    if args.style_conditioning == 'random':
        return torch.randint(0, 4, (batch_size,), device=device, dtype=torch.long)

    if args.style_conditioning == 'none':
        return None

    raise ValueError(f'unknown style_conditioning: {args.style_conditioning}')



def text_width_regularizer(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    text_id: int = TEXT_ID,
    min_w: float = 0.12,
    max_w: float = 0.50,
    edge_margin: float = 0.0,
    too_narrow_weight: float = 35.0,
    too_wide_weight: float = 120.0,
):
    """Regularize text width so text bars do not collapse or span the full canvas.

    bbox_fake is in (cx, cy, w, h), normalized to [0, 1].
    Only text tokens are penalized.
    """
    device = bbox_fake.device
    text_mask = (label == text_id) & (~padding_mask)
    if not text_mask.any():
        return torch.tensor(0.0, device=device)

    text_boxes = bbox_fake[text_mask]
    text_w = text_boxes[:, 2]
    text_cx = text_boxes[:, 0]

    loss_too_narrow = torch.relu(min_w - text_w)
    loss_too_wide = torch.relu(text_w - max_w)
    loss = (loss_too_narrow.mean() * too_narrow_weight) + (loss_too_wide.mean() * too_wide_weight)

    if edge_margin > 0:
        x1 = text_cx - 0.5 * text_w
        x2 = text_cx + 0.5 * text_w
        loss_edge = torch.relu(edge_margin - x1).mean() + torch.relu(x2 - (1.0 - edge_margin)).mean()
        loss = loss + 20.0 * loss_edge

    return loss



def text_geometry_regularizer(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    text_id: int = TEXT_ID,
    min_h: float = 0.028,
    min_area: float = 0.006,
    max_aspect: float = 10.0,
):
    """Keep text boxes readable instead of collapsing into thin lines."""
    device = bbox_fake.device
    text_mask = (label == text_id) & (~padding_mask)
    if not text_mask.any():
        return torch.tensor(0.0, device=device)

    tb = bbox_fake[text_mask]
    w = tb[:, 2].clamp_min(1e-6)
    h = tb[:, 3].clamp_min(1e-6)
    area = w * h
    aspect = w / h

    loss_h = torch.relu(min_h - h).mean()
    loss_area = torch.relu(min_area - area).mean()
    loss_aspect = torch.relu(aspect - max_aspect).mean()

    return (3.0 * loss_h) + (2.0 * loss_area) + (1.0 * loss_aspect)


def image_geometry_regularizer(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    img_id: int = IMG_ID,
    min_area: float = 0.12,
    min_short_side: float = 0.18,
    min_aspect: float = 0.35,
    max_aspect: float = 2.8,
):
    """Avoid image boxes becoming tiny or overly thin strips."""
    device = bbox_fake.device
    img_mask = (label == img_id) & (~padding_mask)
    if not img_mask.any():
        return torch.tensor(0.0, device=device)

    ib = bbox_fake[img_mask]
    w = ib[:, 2].clamp_min(1e-6)
    h = ib[:, 3].clamp_min(1e-6)
    area = w * h
    short_side = torch.minimum(w, h)
    aspect = w / h

    loss_area = torch.relu(min_area - area).mean()
    loss_short = torch.relu(min_short_side - short_side).mean()
    loss_aspect_hi = torch.relu(aspect - max_aspect).mean()
    loss_aspect_lo = torch.relu(min_aspect - aspect).mean()

    return (2.5 * loss_area) + (2.0 * loss_short) + (1.0 * loss_aspect_hi) + (1.0 * loss_aspect_lo)


def batch_shape_prior_loss(
    bbox_fake: torch.Tensor,
    bbox_real: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    target_ids=(TEXT_ID, IMG_ID, SVG_ID),
):
    """Match fake element size distribution to GT within the same batch."""
    device = bbox_fake.device
    loss = torch.tensor(0.0, device=device)
    count = 0

    for tid in target_ids:
        m = (label == tid) & (~padding_mask)
        if int(m.sum().item()) < 2:
            continue

        wf = bbox_fake[..., 2][m].clamp_min(1e-6)
        hf = bbox_fake[..., 3][m].clamp_min(1e-6)
        wr = bbox_real[..., 2][m].clamp_min(1e-6)
        hr = bbox_real[..., 3][m].clamp_min(1e-6)

        med_log_w_fake = torch.median(torch.log(wf))
        med_log_h_fake = torch.median(torch.log(hf))
        med_log_w_real = torch.median(torch.log(wr))
        med_log_h_real = torch.median(torch.log(hr))

        loss = loss + F.l1_loss(med_log_w_fake, med_log_w_real) + F.l1_loss(med_log_h_fake, med_log_h_real)
        count += 1

    return loss / count if count > 0 else loss



def whitespace_style_loss(bbox_fake, label, padding_mask, wr_min=0.6, w_style=1.0, style_mode="max", style_tags=None):
    """
    多樣化留白風格分數（越小越好）：
      - style_mode="max": 舊行為，取 4 種風格中最容易達到的那個（容易導致風格單一）
      - style_mode="random": 每個 sample 隨機指定 1 種風格當目標（更容易學到多樣化留白）

    style_tags: 可選 LongTensor [B]，值域 {0,1,2,3}，分別代表：
      0 frame / 1 side / 2 top-bottom / 3 hybrid
    """
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    losses = []
    margin = 0.05  # wr_min 緩衝區

    if style_tags is not None:
        style_tags = style_tags.to(device)

    for b in range(B):
        valid = ~padding_mask[b]
        # 只用排版元件計算留白風格：排除 Face / BG / Mask
        valid = valid & (label[b] != FACE_ID) & (label[b] != BG_ID) & (label[b] != MASK_ID)
        if valid.sum() == 0:
            continue

        x1, y1, x2, y2 = xywh_to_xyxy(bbox_fake[b][valid]).unbind(-1)
        bw, bh = (x2 - x1).clamp(min=0.0), (y2 - y1).clamp(min=0.0)
        wr = 1.0 - (bw * bh).sum().clamp(0.0, 1.0)  # whitespace ratio

        # 先把「留白比例不夠」的情況拉上來
        if wr < (wr_min - margin):
            losses.append(w_style * (wr_min - margin - wr) ** 2)
            continue

        L = x1.min().clamp(0.001, 0.999)
        R = (1.0 - x2.max()).clamp(0.001, 0.999)
        T = y1.min().clamp(0.001, 0.999)
        Bm = (1.0 - y2.max()).clamp(0.001, 0.999)

        margins = torch.stack([L, R, T, Bm])

        # 4 種留白風格打分（0~1，越大越好）
        S_frame = (margins.mean() - 0.5 * (margins.std(unbiased=False) + 1e-6)).clamp(0.0, 1.0)
        h_max = torch.stack([L, R]).max()
        v_max = torch.stack([T, Bm]).max()
        S_side = (h_max + 0.8 * (h_max - torch.stack([L, R]).min()) + 0.2 * torch.stack([T, Bm]).mean()).clamp(0, 1)
        S_tb = (v_max + 0.8 * (v_max - torch.stack([T, Bm]).min()) + 0.2 * torch.stack([L, R]).mean()).clamp(0, 1)
        S_hybrid = torch.sqrt(S_side * S_tb).clamp(0, 1)

        S_all = torch.stack([S_frame, S_side, S_tb, S_hybrid])

        if style_mode == "random":
            if style_tags is None:
                k = torch.randint(0, 4, (1,), device=device).item()
            else:
                k = int(style_tags[b].item())
            S_style = S_all[k]
        else:
            # 舊行為：取 max（最容易達成的風格）
            S_style = S_all.max()

        losses.append(w_style * (1.0 - S_style))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
def containment_loss(bbox_fake, label, padding_mask, outer_id=IMG_ID, inner_id=FACE_ID, lambda_cont=100.0):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total_loss, count = torch.tensor(0.0, device=device), 0

    for b in range(B):
        valid = ~padding_mask[b]
        out_m = (label[b] == outer_id) & valid
        in_m = (label[b] == inner_id) & valid

        if (not out_m.any()) or (not in_m.any()):
            continue

        out_xyxy = xywh_to_xyxy(bbox_fake[b][out_m]) 
        in_xyxy  = xywh_to_xyxy(bbox_fake[b][in_m])  

        # 距離懲罰 (原本的邏輯)
        p1 = torch.relu(out_xyxy[:, 0:1] - in_xyxy[:, 0].T)
        p2 = torch.relu(in_xyxy[:, 2].T - out_xyxy[:, 2:3])
        p3 = torch.relu(out_xyxy[:, 1:2] - in_xyxy[:, 1].T)
        p4 = torch.relu(in_xyxy[:, 3].T - out_xyxy[:, 3:4])
        dist_penalty = (p1 + p2 + p3 + p4).min(dim=0)[0].mean()

        # 加強版：面積包含懲罰 (修正報錯位置)
        in_area = box_area_xyxy(in_xyxy)
        inter_area = box_intersection_area_xyxy(out_xyxy, in_xyxy) # [M, F]
        contain_ratio = (inter_area / in_area).max(dim=0)[0] # 找最包含的那個 Image
        ratio_penalty = torch.relu(0.98 - contain_ratio).mean() # 要求 98% 面積必須在內

        total_loss += (dist_penalty + ratio_penalty * 5.0) 
        count += 1

    return lambda_cont * (total_loss / count) if count > 0 else total_loss



# 強化對齊損失：將元件吸附到「欄線/行線」(grid lines)，避免用 std 造成全部塌縮成同一條線
def grid_alignment_loss_xy(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    lambda_grid: float = 10.0,
    target_ids=(SVG_ID, TEXT_ID, IMG_ID),
    grid_x=None,
    grid_y=None,
    edge_weight: float = 1.0,
    center_weight: float = 1.0,
    use_right_bottom: bool = False,
):
    """
    bbox_fake: [B, N, 4] in cx,cy,w,h (0~1)
    label:     [B, N]
    padding_mask: [B, N] True = padding
    """
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape

    # 預設 grid：避免把元素強硬推到 0 或 1，保留邊界留白
    if grid_x is None:
        grid_x = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=device)
    else:
        grid_x = torch.as_tensor(grid_x, device=device, dtype=bbox_fake.dtype)
    if grid_y is None:
        grid_y = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=device)
    else:
        grid_y = torch.as_tensor(grid_y, device=device, dtype=bbox_fake.dtype)

    losses = []
    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() == 0:
            continue

        tgt = torch.zeros_like(valid, dtype=torch.bool)
        for tid in target_ids:
            tgt = tgt | (label[b] == tid)
        tgt &= valid

        if tgt.sum() == 0:
            continue

        xyxy = xywh_to_xyxy(bbox_fake[b][tgt])
        x1, y1, x2, y2 = xyxy.unbind(-1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # 吸附到最近 grid line
        dx_edge = torch.abs(x1[:, None] - grid_x[None, :]).min(dim=1)[0]
        dy_edge = torch.abs(y1[:, None] - grid_y[None, :]).min(dim=1)[0]
        dx_center = torch.abs(cx[:, None] - grid_x[None, :]).min(dim=1)[0]
        dy_center = torch.abs(cy[:, None] - grid_y[None, :]).min(dim=1)[0]

        if use_right_bottom:
            dx_edge2 = torch.abs(x2[:, None] - grid_x[None, :]).min(dim=1)[0]
            dy_edge2 = torch.abs(y2[:, None] - grid_y[None, :]).min(dim=1)[0]
            dx_edge = 0.5 * (dx_edge + dx_edge2)
            dy_edge = 0.5 * (dy_edge + dy_edge2)

        loss_b = edge_weight * (dx_edge.mean() + dy_edge.mean()) + center_weight * (dx_center.mean() + dy_center.mean())
        losses.append(loss_b)

    if not losses:
        return torch.tensor(0.0, device=device)

    return lambda_grid * torch.stack(losses).mean()


def element_size_loss(bbox_fake, label, padding_mask, target_ids, max_area, lambda_size=10.0):
    device = bbox_fake.device
    mask = torch.zeros_like(label, dtype=torch.bool)
    for tid in target_ids: mask = mask | (label == tid)
    mask &= (~padding_mask)
    if not mask.any(): return torch.tensor(0.0, device=device)
    areas = bbox_fake[:, :, 2] * bbox_fake[:, :, 3]
    return lambda_size * torch.relu(areas[mask] - max_area).mean()

def face_coverage_loss(bbox_fake, label, padding_mask, lambda_face=1.0):
    """
    懲罰生成的「文字/SVG」覆蓋到生成的「人臉」。
    """
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total, count = torch.tensor(0.0, device=device), 0

    for b in range(B):
        valid = ~padding_mask[b]
        f_m = (label[b] == FACE_ID) & valid
        # 懲罰文字(1)或小型SVG(0)壓在臉上
        other_m = ( (label[b] == TEXT_ID) | (label[b] == SVG_ID) ) & valid
        
        if not (f_m.any() and other_m.any()):
            continue

        fb_xyxy = xywh_to_xyxy(bbox_fake[b][f_m])
        ob_xyxy = xywh_to_xyxy(bbox_fake[b][other_m])

        inter = box_intersection_area_xyxy(ob_xyxy, fb_xyxy) # [K, F]
        f_area = box_area_xyxy(fb_xyxy)
        
        cover = (inter.sum(dim=0) / f_area).clamp(0, 1)
        total += cover.mean()
        count += 1

    return lambda_face * (total / count) if count > 0 else total


def face_occlusion_stats(bbox, label, padding_mask, thr=0.01):
    """回傳 (平均遮擋比例, 被遮擋樣本比例)。

    遮擋定義：文字(TEXT_ID)、SVG(SVG_ID)、遮罩(MASK_ID) 與人臉(FACE_ID) 的交集面積 / 人臉面積。
    注意：不把 Image(IMG_ID) 與 Background(BG_ID) 視為遮擋者，因為 Face 本來就在 Image 內。
    """
    device = bbox.device
    B, N, _ = bbox.shape
    ratios = []
    occ_flags = []
    for b in range(B):
        valid = ~padding_mask[b]
        f_m = (label[b] == FACE_ID) & valid
        o_m = ((label[b] == TEXT_ID) | (label[b] == SVG_ID) | (label[b] == MASK_ID)) & valid
        if not (f_m.any() and o_m.any()):
            ratios.append(torch.tensor(0.0, device=device))
            occ_flags.append(torch.tensor(0.0, device=device))
            continue

        fb = xywh_to_xyxy(bbox[b][f_m])
        ob = xywh_to_xyxy(bbox[b][o_m])
        inter = box_intersection_area_xyxy(ob, fb)   # [K,F]
        f_area = box_area_xyxy(fb).clamp_min(1e-8)   # [F]

        cover_per_face = (inter.sum(dim=0) / f_area).clamp(0, 1)
        ratios.append(cover_per_face.mean())
        occ_flags.append((cover_per_face > thr).any().float())

    return torch.stack(ratios).mean(), torch.stack(occ_flags).mean()


def pairwise_overlap_loss(bbox_fake, label, padding_layout, big_img_thr: float = 0.90):
    """Pairwise overlap penalty with special rule for huge images.

    - Image-Image overlap is penalized.
    - BUT if either image covers >= big_img_thr (e.g., 0.90) of the canvas area,
      we allow Image-Image overlap (treat it as background-like / full-bleed element).
    - BG / MASK are ignored (weight=0), and IMG-FACE overlap is ignored (face is inside image).
    """
    padding_mask = padding_layout
    device = bbox_fake.device

    # label-pair weights
    W = torch.ones((10, 10), device=device)
    W[BG_ID, :], W[:, BG_ID], W[MASK_ID, :], W[:, MASK_ID] = 0.0, 0.0, 0.0, 0.0
    W[IMG_ID, FACE_ID], W[FACE_ID, IMG_ID] = 0.0, 0.0
    W[IMG_ID, IMG_ID] = 4.0

    # 強化你真正在意的物件不重疊
    W[TEXT_ID, TEXT_ID] = 20.0
    W[TEXT_ID, FACE_ID], W[FACE_ID, TEXT_ID] = 25.0, 25.0
    W[TEXT_ID, IMG_ID], W[IMG_ID, TEXT_ID] = 12.0, 12.0
    W[TEXT_ID, SVG_ID], W[SVG_ID, TEXT_ID] = 14.0, 14.0
    W[SVG_ID, SVG_ID] = 6.0
    W[IMG_ID, SVG_ID], W[SVG_ID, IMG_ID] = 4.0, 4.0

    losses = []
    for b in range(bbox_fake.size(0)):
        valid = ~padding_mask[b]
        if valid.sum() <= 1:
            continue

        xyxy = xywh_to_xyxy(bbox_fake[b][valid])
        labs = label[b][valid]

        inter = box_intersection_area_xyxy(xyxy, xyxy)
        n = inter.size(0)
        eye = torch.eye(n, device=inter.device, dtype=inter.dtype)
        inter = inter * (1.0 - eye)

        area = box_area_xyxy(xyxy, eps=1e-4)
        denom = torch.minimum(area[:, None], area[None, :]).clamp(min=1e-4)
        overlap_ratio = (inter / denom).clamp(max=2.0)

        # ---- special rule: allow IMG-IMG overlap if either IMG is huge (>= 90% canvas) ----
        is_img = (labs == IMG_ID)
        is_big_img = is_img & (area >= big_img_thr)
        allow_img_img = (is_img[:, None] & is_img[None, :]) & (is_big_img[:, None] | is_big_img[None, :])
        overlap_ratio = overlap_ratio * (~allow_img_img).to(overlap_ratio.dtype)

        weights = W[labs[:, None], labs[None, :]]
        loss_b = (overlap_ratio * weights).triu(1).sum()
        losses.append(loss_b / ((valid.sum() * (valid.sum() - 1)) / 2).clamp(min=1))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def local_spacing_loss(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_layout: torch.Tensor,
    min_gap: float = 0.04,
    big_img_thr: float = 0.90,
):
    """Penalize pairs that are too close even when they do not overlap.

    This complements pairwise_overlap_loss:
      - overlap_loss handles actual intersection
      - local_spacing_loss keeps text / svg / image blocks from hugging each other
    """
    device = bbox_fake.device
    padding_mask = padding_layout

    G = torch.zeros((10, 10), device=device)
    G[TEXT_ID, TEXT_ID] = 1.60
    G[TEXT_ID, IMG_ID] = G[IMG_ID, TEXT_ID] = 1.40
    G[TEXT_ID, SVG_ID] = G[SVG_ID, TEXT_ID] = 1.30
    G[SVG_ID, SVG_ID] = 0.90
    G[IMG_ID, IMG_ID] = 0.45
    G[IMG_ID, SVG_ID] = G[SVG_ID, IMG_ID] = 0.70

    losses = []
    for b in range(bbox_fake.size(0)):
        valid = ~padding_mask[b]
        if valid.sum() <= 1:
            continue

        xyxy = xywh_to_xyxy(bbox_fake[b][valid])
        labs = label[b][valid]
        x1, y1, x2, y2 = xyxy.unbind(-1)
        n = xyxy.size(0)

        dx = torch.maximum(x1[None, :] - x2[:, None], x1[:, None] - x2[None, :]).clamp(min=0.0)
        dy = torch.maximum(y1[None, :] - y2[:, None], y1[:, None] - y2[None, :]).clamp(min=0.0)
        edge_gap = torch.sqrt(dx * dx + dy * dy + 1e-12)

        inter = box_intersection_area_xyxy(xyxy, xyxy)
        eye = torch.eye(n, device=device, dtype=inter.dtype)
        inter = inter * (1.0 - eye)
        non_overlap = (inter <= 1e-8).to(edge_gap.dtype)

        area = box_area_xyxy(xyxy, eps=1e-4)
        is_img = (labs == IMG_ID)
        is_big_img = is_img & (area >= big_img_thr)
        allow_img_img = (is_img[:, None] & is_img[None, :]) & (is_big_img[:, None] | is_big_img[None, :])

        weights = G[labs[:, None], labs[None, :]]
        gap_penalty = torch.relu(min_gap - edge_gap)
        pair_penalty = gap_penalty * weights * non_overlap * (~allow_img_img).to(gap_penalty.dtype)

        num_pairs = (n * (n - 1)) / 2
        losses.append(pair_penalty.triu(1).sum() / max(num_pairs, 1))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def poster_template_loss(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_layout: torch.Tensor,
    template: str = "center_vertical",
    lambda_template: float = 20.0,
):
    """Soft poster-template constraints for several poster layout families.

    The goal is not to force a fixed token order, but to bias the overall
    geometric distribution toward a chosen poster template.
    """
    device = bbox_fake.device
    losses = []

    for b in range(bbox_fake.size(0)):
        valid = ~padding_layout[b]
        if valid.sum() == 0:
            continue

        boxes = bbox_fake[b]
        text_m = (label[b] == TEXT_ID) & valid
        img_m = (label[b] == IMG_ID) & valid
        svg_m = (label[b] == SVG_ID) & valid
        graphic_m = img_m | svg_m

        xyxy_all = xywh_to_xyxy(boxes[valid])
        x1_all, y1_all, x2_all, y2_all = xyxy_all.unbind(-1)

        loss_b = torch.tensor(0.0, device=device)

        if template == "center_vertical":
            valid_boxes = boxes[valid]
            if valid_boxes.size(0) > 1:
                order = torch.argsort(valid_boxes[:, 1])
                sorted_xyxy = xywh_to_xyxy(valid_boxes[order])
                gaps = sorted_xyxy[1:, 1] - sorted_xyxy[:-1, 3]
                loss_b += torch.relu(0.035 - gaps).mean() * 2.0

            if text_m.any():
                loss_b += ((boxes[text_m][:, 0] - 0.50) ** 2).mean() * 2.5
                loss_b += torch.relu(boxes[text_m][:, 2] - 0.38).mean() * 1.2

            if graphic_m.any():
                loss_b += ((boxes[graphic_m][:, 0] - 0.50) ** 2).mean() * 1.2

            left_margin = x1_all.min()
            right_margin = 1.0 - x2_all.max()
            loss_b += ((left_margin - right_margin) ** 2) * 1.5

        elif template == "right_column":
            if text_m.any():
                loss_b += ((boxes[text_m][:, 0] - 0.76) ** 2).mean() * 2.8
                loss_b += torch.relu(boxes[text_m][:, 2] - 0.34).mean() * 1.0

            if graphic_m.any():
                loss_b += ((boxes[graphic_m][:, 0] - 0.72) ** 2).mean() * 0.8

            loss_b += torch.relu(0.30 - x1_all.min()) * 3.0
            right_margin = 1.0 - x2_all.max()
            loss_b += torch.relu(0.04 - right_margin) * 2.0

        elif template == "top_hero":
            if img_m.any():
                img_boxes = boxes[img_m]
                img_areas = img_boxes[:, 2] * img_boxes[:, 3]
                k = torch.argmax(img_areas)
                main_img = img_boxes[k]
                main_area = img_areas[k]

                loss_b += ((main_img[0] - 0.50) ** 2) * 1.2
                loss_b += ((main_img[1] - 0.64) ** 2) * 2.0
                loss_b += torch.relu(0.20 - main_area) * 4.0

            if text_m.any():
                cy = boxes[text_m][:, 1]
                d_top = torch.abs(cy - 0.14)
                d_mid = torch.abs(cy - 0.43)
                loss_b += torch.minimum(d_top, d_mid).mean() * 1.2

            loss_b += torch.relu(0.12 - y1_all.min()) * 1.5

        elif template == "bottom_caption":
            if img_m.any():
                img_boxes = boxes[img_m]
                img_areas = img_boxes[:, 2] * img_boxes[:, 3]
                k = torch.argmax(img_areas)
                main_img = img_boxes[k]
                main_area = img_areas[k]

                loss_b += ((main_img[0] - 0.50) ** 2) * 1.0
                loss_b += ((main_img[1] - 0.42) ** 2) * 2.0
                loss_b += torch.relu(0.18 - main_area) * 4.0

            if text_m.any():
                loss_b += torch.relu(0.68 - boxes[text_m][:, 1]).mean() * 2.5
        
        elif template == "hybrid":
            # 這裡先把 hybrid 定義成：偏右下聚集，保留左上/上方/左方較大留白
            if img_m.any():
                img_boxes = boxes[img_m]
                img_areas = img_boxes[:, 2] * img_boxes[:, 3]
                k = torch.argmax(img_areas)
                main_img = img_boxes[k]
                main_area = img_areas[k]

                loss_b += ((main_img[0] - 0.68) ** 2) * 1.4
                loss_b += ((main_img[1] - 0.66) ** 2) * 1.6
                loss_b += torch.relu(0.18 - main_area) * 4.0

            if text_m.any():
                # 文字也不要貼左上，偏右側/中下
                loss_b += ((boxes[text_m][:, 0] - 0.72) ** 2).mean() * 2.0
                loss_b += ((boxes[text_m][:, 1] - 0.38) ** 2).mean() * 1.2

            # 明確要求左、上都留比較大 margin
            loss_b += torch.relu(0.18 - x1_all.min()) * 3.0
            loss_b += torch.relu(0.18 - y1_all.min()) * 3.0

        losses.append(loss_b)

    if not losses:
        return torch.tensor(0.0, device=device)
    return lambda_template * torch.stack(losses).mean()




def center_vertical_role_loss(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_layout: torch.Tensor,
):
    """Role-aware template loss specialized for center_vertical posters."""
    device = bbox_fake.device
    losses = []

    for b in range(bbox_fake.size(0)):
        valid = ~padding_layout[b]
        text_m = (label[b] == TEXT_ID) & valid
        img_m = (label[b] == IMG_ID) & valid

        if not img_m.any():
            continue

        boxes = bbox_fake[b]
        img_boxes = boxes[img_m]
        img_areas = img_boxes[:, 2] * img_boxes[:, 3]
        k = torch.argmax(img_areas)
        main_img = img_boxes[k]

        loss_b = torch.tensor(0.0, device=device)
        loss_b += ((main_img[0] - 0.50) ** 2) * 2.0
        loss_b += torch.relu(0.14 - (main_img[2] * main_img[3])) * 3.0

        if text_m.any():
            t = boxes[text_m]
            loss_b += ((t[:, 0] - 0.50) ** 2).mean() * 2.0
            loss_b += torch.relu(0.16 - t[:, 2]).mean() * 2.0
            loss_b += torch.relu(t[:, 2] - 0.42).mean() * 1.5

            img_top = main_img[1] - main_img[3] / 2.0
            img_bot = main_img[1] + main_img[3] / 2.0
            text_top = t[:, 1] - t[:, 3] / 2.0
            text_bot = t[:, 1] + t[:, 3] / 2.0

            overlap_y = torch.minimum(text_bot, img_bot.expand_as(text_bot)) - \
                        torch.maximum(text_top, img_top.expand_as(text_top))
            overlap_y = overlap_y.clamp(min=0.0)
            loss_b += overlap_y.mean() * 2.0

        losses.append(loss_b)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()



def _content_valid_mask(label_b: torch.Tensor, valid_b: torch.Tensor) -> torch.Tensor:
    return valid_b & (label_b != FACE_ID) & (label_b != BG_ID) & (label_b != MASK_ID)


def _get_dynamic_region_cfg(template: str):
    """Dynamic content-region targets driven by element count + area_sum.

    不再直接寫死 xmin/xmax/ymin/ymax，而是：
    1) 先給每種 template 一個內容區中心(cx, cy)
    2) 再根據前景元素數量 count 與面積總和 area_sum，動態推估 union box 應有的寬高
    3) 最後只懲罰 union box 太大、太偏、或留白方向不符合 template
    """
    cfg = {
        "center_vertical": {
            "cx": 0.50, "cy": 0.54,
            "base_w": 0.18, "base_h": 0.24,
            "count_w": 0.060, "count_h": 0.085,
            "area_w": 0.75, "area_h": 0.95,
            "min_w": 0.24, "max_w": 0.72,
            "min_h": 0.30, "max_h": 0.88,
            "dir_w": 0.0, "dir_h": 0.0,
            "dir_margin": 0.00,
        },
        "right_column": {
            "cx": 0.72, "cy": 0.54,
            "base_w": 0.16, "base_h": 0.22,
            "count_w": 0.052, "count_h": 0.082,
            "area_w": 0.60, "area_h": 0.92,
            "min_w": 0.22, "max_w": 0.52,
            "min_h": 0.28, "max_h": 0.86,
            "dir_w": +1.0, "dir_h": 0.0,
            "dir_margin": 0.12,
        },
        "top_hero": {
            "cx": 0.52, "cy": 0.62,
            "base_w": 0.20, "base_h": 0.18,
            "count_w": 0.075, "count_h": 0.060,
            "area_w": 0.88, "area_h": 0.68,
            "min_w": 0.28, "max_w": 0.82,
            "min_h": 0.24, "max_h": 0.68,
            "dir_w": 0.0, "dir_h": +1.0,
            "dir_margin": 0.10,
        },
        "bottom_caption": {
            "cx": 0.50, "cy": 0.42,
            "base_w": 0.22, "base_h": 0.20,
            "count_w": 0.070, "count_h": 0.070,
            "area_w": 0.82, "area_h": 0.72,
            "min_w": 0.30, "max_w": 0.84,
            "min_h": 0.26, "max_h": 0.72,
            "dir_w": 0.0, "dir_h": -1.0,
            "dir_margin": 0.08,
        },
        "hybrid": {
            "cx": 0.68, "cy": 0.66,
            "base_w": 0.14, "base_h": 0.16,
            "count_w": 0.060, "count_h": 0.060,
            "area_w": 0.56, "area_h": 0.60,
            "min_w": 0.22, "max_w": 0.56,
            "min_h": 0.24, "max_h": 0.60,
            "dir_w": +1.0, "dir_h": +1.0,
            "dir_margin": 0.14,
        },
    }
    return cfg[template]



def content_region_loss(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_layout: torch.Tensor,
    template: str = "center_vertical",
    lambda_region: float = 10.0,
    center_weight: float = 1.0,
    compact_weight: float = 1.0,
):
    """Dynamic union-box region loss based on count + area_sum.

    舊版問題：
      - 直接把 union box 壓到 template 寫死邊界內
      - 元素少/多、面積小/大，都用同一組 xmax/ymax/max_w/max_h

    新版做法：
      - 用前景元素數量 count
      - 用前景面積總和 area_sum
      - 動態決定 union box 應該允許的 target_w / target_h
      - template 只保留「中心位置」與「留白方向偏好」
    """
    device = bbox_fake.device
    losses = []
    cfg = _get_dynamic_region_cfg(template)

    for b in range(bbox_fake.size(0)):
        valid = ~padding_layout[b]
        valid = _content_valid_mask(label[b], valid)
        if valid.sum() == 0:
            continue

        boxes = bbox_fake[b][valid]
        xyxy = xywh_to_xyxy(boxes)
        x1 = xyxy[:, 0].min()
        y1 = xyxy[:, 1].min()
        x2 = xyxy[:, 2].max()
        y2 = xyxy[:, 3].max()

        cw = (x2 - x1).clamp(min=0.0)
        ch = (y2 - y1).clamp(min=0.0)
        ccx = (x1 + x2) / 2.0
        ccy = (y1 + y2) / 2.0

        count = valid.sum().float()
        area_sum = (boxes[:, 2].clamp(min=0.0) * boxes[:, 3].clamp(min=0.0)).sum()
        area_term = torch.sqrt(area_sum.clamp(min=1e-8))

        target_w = cfg["base_w"] + cfg["count_w"] * (count - 1.0) + cfg["area_w"] * area_term
        target_h = cfg["base_h"] + cfg["count_h"] * (count - 1.0) + cfg["area_h"] * area_term
        target_w = target_w.clamp(cfg["min_w"], cfg["max_w"])
        target_h = target_h.clamp(cfg["min_h"], cfg["max_h"])

        left = x1
        right = 1.0 - x2
        top = y1
        bottom = 1.0 - y2

        loss_b = torch.tensor(0.0, device=device)

        # 1) union center 靠近 template 中心
        loss_b += center_weight * ((ccx - cfg["cx"]) ** 2 + (ccy - cfg["cy"]) ** 2)

        # 2) union 尺寸不要超過「count + area_sum 推出來的合理大小」
        loss_b += compact_weight * torch.relu(cw - target_w)
        loss_b += compact_weight * torch.relu(ch - target_h)

        # 3) 避免退化成過小內容區：只給弱約束，不把內容硬撐大
        loss_b += 0.25 * compact_weight * torch.relu(0.55 * target_w - cw)
        loss_b += 0.25 * compact_weight * torch.relu(0.55 * target_h - ch)

        # 4) template 的留白方向偏好，不再是固定邊界，而是 margin 關係
        # dir_w = +1 代表左邊留白應該 > 右邊；dir_w = -1 反之
        # dir_h = +1 代表上方留白應該 > 下方；dir_h = -1 反之
        if cfg["dir_w"] > 0:
            loss_b += torch.relu((right + cfg["dir_margin"]) - left)
        elif cfg["dir_w"] < 0:
            loss_b += torch.relu((left + cfg["dir_margin"]) - right)

        if cfg["dir_h"] > 0:
            loss_b += torch.relu((bottom + cfg["dir_margin"]) - top)
        elif cfg["dir_h"] < 0:
            loss_b += torch.relu((top + cfg["dir_margin"]) - bottom)

        losses.append(loss_b)

    if not losses:
        return torch.tensor(0.0, device=device)
    return lambda_region * torch.stack(losses).mean()



def content_union_stats(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_layout: torch.Tensor,
):
    """Diagnostics for union box + the new dynamic-region drivers.

    回傳：
      union_w_mean, union_h_mean, union_x1_mean, union_y1_mean,
      count_mean, area_sum_mean
    """
    device = bbox_fake.device
    union_w, union_h = [], []
    union_x1, union_y1 = [], []
    elem_count, area_sum = [], []

    for b in range(bbox_fake.size(0)):
        valid = ~padding_layout[b]
        valid = _content_valid_mask(label[b], valid)
        if valid.sum() == 0:
            continue

        boxes = bbox_fake[b][valid]
        xyxy = xywh_to_xyxy(boxes)
        x1 = xyxy[:, 0].min()
        y1 = xyxy[:, 1].min()
        x2 = xyxy[:, 2].max()
        y2 = xyxy[:, 3].max()

        union_w.append((x2 - x1).clamp(min=0.0))
        union_h.append((y2 - y1).clamp(min=0.0))
        union_x1.append(x1)
        union_y1.append(y1)
        elem_count.append(valid.sum().float())
        area_sum.append((boxes[:, 2].clamp(min=0.0) * boxes[:, 3].clamp(min=0.0)).sum())

    if len(union_w) == 0:
        z = torch.tensor(0.0, device=device)
        return z, z, z, z, z, z

    return (
        torch.stack(union_w).mean(),
        torch.stack(union_h).mean(),
        torch.stack(union_x1).mean(),
        torch.stack(union_y1).mean(),
        torch.stack(elem_count).mean(),
        torch.stack(area_sum).mean(),
    )

def face_anchor_loss_relative(bbox_fake, label, padding_mask, face_rel, face_mask, face2img, lambda_anchor=50.0):
    """
    Relative face-anchor loss:
    - Target (GT) is relative to its corresponding image box (rx, ry, rw, rh).
    - Prediction compares the generated face box relative to the generated image box.

    face_rel:  (B, max_faces, 4)   [rx, ry, rw, rh]
    face2img:  (B, max_faces)      token index of corresponding image in the SAME sequence
    face_mask: (B, max_faces)      True for valid GT faces
    """
    device = bbox_fake.device
    B = bbox_fake.size(0)
    total_loss = torch.tensor(0.0, device=device)
    total_count = 0
    eps = 1e-6

    for b in range(B):
        valid = ~padding_mask[b]

        # indices of face tokens produced by generator in this sample
        f_idx = torch.where((label[b] == FACE_ID) & valid)[0]
        n_gt = int(face_mask[b].sum().item())
        n = min(int(f_idx.numel()), n_gt)
        if n <= 0:
            continue

        # build r_pred and r_gt for valid (face,image) pairs
        r_pred_list = []
        r_gt_list = []

        for i in range(n):
            img_idx = int(face2img[b, i].item())
            if img_idx < 0:
                continue
            if img_idx >= label.size(1):
                continue
            if (not valid[img_idx].item()) or (label[b, img_idx].item() != IMG_ID):
                continue

            img_box = bbox_fake[b, img_idx]
            face_box = bbox_fake[b, f_idx[i]]

            w_img = img_box[2].clamp_min(eps)
            h_img = img_box[3].clamp_min(eps)

            rx = (face_box[0] - img_box[0]) / w_img
            ry = (face_box[1] - img_box[1]) / h_img
            rw = face_box[2] / w_img
            rh = face_box[3] / h_img

            r_pred_list.append(torch.stack([rx, ry, rw, rh], dim=0))
            r_gt_list.append(face_rel[b, i])

        if len(r_pred_list) == 0:
            continue

        r_pred = torch.stack(r_pred_list, dim=0)
        r_gt = torch.stack(r_gt_list, dim=0)
        total_loss += F.mse_loss(r_pred, r_gt)
        total_count += 1

    return lambda_anchor * (total_loss / total_count) if total_count > 0 else total_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='/home/albee/const_layout_whitespace/pretrained/layoutnet_crello.pth.tar')
    parser.add_argument('--name', type=str, default='v13_ft')
    parser.add_argument('--occ_thresh', type=float, default=0.01)
    parser.add_argument('--dataset', type=str, default='crello')
    parser.add_argument('--mix_weights', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=('W_CRELLO', 'W_HQ', 'W_WS60'),
                        help='三個固定 pkl 的抽樣權重：crello.pkl、high_quality.pkl、60%_ws.pkl，例如 0.4 0.3 0.3')
    parser.add_argument('--mix_epoch_size', type=int, default=50000,
                        help='混合資料集每個 epoch 視為多少筆樣本')
    parser.add_argument('--single_pkl', type=int, default=0,
                        help='1=不要混合，僅使用 PKL_PATH_WS_60；0=使用三個固定 pkl 混合訓練')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_nodes', type=int, default=5)
    parser.add_argument('--max_faces', type=int, default=4)
    parser.add_argument('--iteration', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--lambda_ws', type=float, default=0.08)
    parser.add_argument('--wr_min', type=float, default=0.65)
    parser.add_argument('--lambda_ov', type=float, default=1.5)
    parser.add_argument('--lambda_face', type=float, default=1.0)
    parser.add_argument('--lambda_align', type=float, default=10.0)   # 對齊 (grid) 權重
    parser.add_argument('--align_start', type=int, default=8000)      # 幾步後開始打開對齊
    parser.add_argument('--align_warmup', type=int, default=5000)     # 對齊 warmup 步數
    parser.add_argument('--lambda_cont', type=float, default=50.0)    # containment (face in image) 權重
    parser.add_argument('--cont_start', type=int, default=15000)
    parser.add_argument('--cont_warmup', type=int, default=5000)
    parser.add_argument('--lambda_anchor', type=float, default=30.0)  # face anchor 權重
    parser.add_argument('--anchor_start', type=int, default=15000)
    parser.add_argument('--anchor_warmup', type=int, default=5000)
    parser.add_argument('--face_start', type=int, default=12000)      # face coverage 開始步
    parser.add_argument('--face_warmup', type=int, default=5000)
    parser.add_argument('--lambda_style', type=float, default=30.0)   # 你前面 style_loss *150 太兇，改成可調
    parser.add_argument('--layout_template', type=str, default='center_vertical',
                        choices=['top_hero', 'center_vertical', 'right_column', 'bottom_caption', 'hybrid'])
    parser.add_argument('--lambda_template', type=float, default=20.0)
    parser.add_argument('--template_start', type=int, default=3000)
    parser.add_argument('--template_warmup', type=int, default=4000)
    parser.add_argument('--lambda_space', type=float, default=10.0)    # local spacing：避免元素彼此太貼近
    parser.add_argument('--space_min_gap', type=float, default=0.04)   # 期望最小間距
    parser.add_argument('--space_start', type=int, default=5000)
    parser.add_argument('--space_warmup', type=int, default=5000)
    parser.add_argument('--lambda_text_width', type=float, default=6.0)
    parser.add_argument('--text_min_w', type=float, default=0.12)
    parser.add_argument('--text_max_w', type=float, default=0.45)
    parser.add_argument('--text_max_w_top_hero', type=float, default=0.62)
    parser.add_argument('--text_edge_margin', type=float, default=0.03)
    parser.add_argument('--text_width_start', type=int, default=2000)
    parser.add_argument('--text_width_warmup', type=int, default=4000)
    parser.add_argument('--text_too_narrow_weight', type=float, default=35.0)
    parser.add_argument('--text_too_wide_weight', type=float, default=120.0)
    parser.add_argument('--lambda_text_geom', type=float, default=8.0)
    parser.add_argument('--text_min_h', type=float, default=0.028)
    parser.add_argument('--text_min_area', type=float, default=0.006)
    parser.add_argument('--text_max_aspect', type=float, default=10.0)
    parser.add_argument('--lambda_img_geom', type=float, default=8.0)
    parser.add_argument('--img_min_area', type=float, default=0.12)
    parser.add_argument('--img_min_short_side', type=float, default=0.18)
    parser.add_argument('--img_min_aspect', type=float, default=0.35)
    parser.add_argument('--img_max_aspect', type=float, default=2.8)
    parser.add_argument('--lambda_shape_prior', type=float, default=6.0)
    parser.add_argument('--shape_prior_start', type=int, default=4000)
    parser.add_argument('--shape_prior_warmup', type=int, default=4000)
    parser.add_argument('--lambda_role_layout', type=float, default=10.0)
    parser.add_argument('--lambda_region', type=float, default=12.0)
    parser.add_argument('--region_start', type=int, default=4000)
    parser.add_argument('--region_warmup', type=int, default=4000)
    parser.add_argument('--region_center_weight', type=float, default=1.0)
    parser.add_argument('--region_compact_weight', type=float, default=1.0)
    parser.add_argument('--role_layout_start', type=int, default=5000)
    parser.add_argument('--role_layout_warmup', type=int, default=4000)
    parser.add_argument('--fix_aspect', type=int, default=1)           # 1=固定長寬比(只等比縮放), 0=不啟用
    parser.add_argument('--fix_text_aspect', type=int, default=0)      # 0=不要把文字框長寬比硬拉回 GT
    parser.add_argument('--ws_style_mode', type=str, default='random', choices=['max','random'])  # 留白風格選擇方式
    parser.add_argument('--style_conditioning', type=str, default='template', choices=['template', 'random', 'none'],
                        help='template=依 layout_template 指定風格; random=每個 sample 真隨機風格; none=交給 whitespace_style_loss 內部抽樣')
    parser.add_argument('--train_stage', type=str, default='full', choices=['baseline', 'text', 'text_image', 'full'],
                        help='baseline -> text -> text_image -> full 的分階段訓練開關')
    parser.add_argument('--fix_img_prob', type=float, default=0.0)     # 以機率將 Image(以及 Face) 固定到 GT (0~1)
    parser.add_argument('--freeze_face', type=int, default=1)        # 1=Face用GT相對座標貼回當前image(不猜face位置), 0=關閉
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--aug_flip', action='store_true')
    parser.add_argument('--aug_vflip', action='store_true')
    parser.add_argument('--flip_prob', type=float, default=0.5)
    parser.add_argument('--vis_every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fixed_sample', type=str, default='fixed_sample_v49_n5.pt') # 確保檔名正確
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--resume_ckpt', type=str, default='')
    parser.add_argument('--detect_anomaly', type=int, default=0, help='1=debug backward graph, 0=正式訓練關閉以加速')
    parser.add_argument('--G_d_model', type=int, default=256)
    parser.add_argument('--G_nhead', type=int, default=4)
    parser.add_argument('--G_num_layers', type=int, default=4)
    parser.add_argument('--D_d_model', type=int, default=256)
    parser.add_argument('--D_nhead', type=int, default=4)
    parser.add_argument('--D_num_layers', type=int, default=4)
    args = parser.parse_args()

    # ---- Reproducibility / randomness control ----
    # 若你要可重現：指定 --seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # 初始化實驗，取得輸出的資料夾路徑 out_dir
    out_dir = init_experiment(args, "LayoutGAN++")
    writer = SummaryWriter(out_dir)

    # ===== CSV logger for face occlusion =====
    occ_csv = os.path.join(out_dir, "face_occlusion_log.csv")
    if not os.path.exists(occ_csv):
        with open(occ_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "occ_ratio_mean", "occ_rate", "n_face_tokens", "n_occluder_tokens"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 資料載入
    base_dataset = get_dataset(args.dataset, 'train')

    def _build_raw_dataset_from_pkl(pkl_path: str):
        with open(pkl_path, 'rb') as f:
            data_list = pickle.load(f)
        return RawLayoutDataset(
            data_list,
            num_classes=base_dataset.num_classes,
            max_nodes=args.max_nodes,
            max_faces=args.max_faces,
            colors=base_dataset.colors,
            aug_hflip=args.aug_flip,
            aug_vflip=args.aug_vflip,
            flip_prob=args.flip_prob,
        )

    # 預設沿用原本單一資料集行為
    train_dataset = base_dataset

    # 固定三個 pkl 路徑；下參數時只需要給權重
    fixed_mix_pkls = [
        PKL_PATH_CRELLO_FULL,
        PKL_PATH_HIGH_QUALITY,
        PKL_PATH_WS_60,
    ]

    if len(args.mix_weights) != 3:
        raise ValueError(f'--mix_weights 必須剛好給 3 個值，目前收到 {len(args.mix_weights)} 個')

    if args.single_pkl == 1:
        pkl_path = PKL_PATH_WS_60
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f'找不到 pkl: {pkl_path}')
        train_dataset = _build_raw_dataset_from_pkl(pkl_path)
        print(f"[Single PKL] loaded: {pkl_path} (n={len(train_dataset)})")
    else:
        mixed_datasets = []
        for pkl_path in fixed_mix_pkls:
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f'找不到 pkl: {pkl_path}')
            ds = _build_raw_dataset_from_pkl(pkl_path)
            mixed_datasets.append(ds)
            print(f"[MIX] loaded: {pkl_path} (n={len(ds)})")

        train_dataset = WeightedMixedRawDataset(
            mixed_datasets,
            source_weights=args.mix_weights,
            epoch_size=args.mix_epoch_size,
            seed=args.seed,
        )
        print(
            f"[MIX] pkl order=[crello.pkl, high_quality.pkl, crello_train_ws_gt0.6_with_face.pkl] "
            f"weights={train_dataset.source_weights.tolist()} epoch_size={len(train_dataset)}"
        )
    
    # 強制使用標準 RGB 整數，解決 TypeError
    train_dataset.colors = [
        (31, 119, 180),   # 0: Svgelement
        (44, 160, 44),    # 1: Textelement
        (148, 103, 189),  # 2: Imageelement
        (227, 119, 194),  # 3: colorbackground
        (188, 189, 34),   # 4: Svgmaskelement
        (158, 218, 229),  # 5: face 
    ]
    # 再次確保所有數值都是 int
    train_dataset.colors = [tuple(int(c) for c in color) for color in train_dataset.colors]

    # 確保這行在所有路徑下都會執行
    dl_gen = None
    if args.seed is not None:
        dl_gen = torch.Generator()
        dl_gen.manual_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )
    # 模型初始化
    netG = Generator(args.latent_size, train_dataset.num_classes, d_model=args.G_d_model, nhead=args.G_nhead, num_layers=args.G_num_layers).to(device)
    netD = Discriminator(train_dataset.num_classes, d_model=args.D_d_model, nhead=args.D_nhead, num_layers=args.D_num_layers).to(device)

    # 關鍵：從 layoutnet_crello.pth.tar 載入預訓練權重
    pretrained_path = "/home/albee/const_layout_whitespace/pretrained/layoutnet_crello.pth.tar"
    # === 修正後的載入區塊 ===
    # === 修正後的載入區塊 (替代原本 netG 初始化之後的邏輯) ===
    if args.pretrained:
        print(f"===> 正在進行【精準路徑轉譯】載入預訓練權重...")
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        old_sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        curr_sd = netG.state_dict()
        new_state_dict = {}

        for k, v in old_sd.items():
            tk = k.replace('enc_transformer.core.', 'transformer.')
            tk = tk.replace('dec_transformer.core.', 'transformer.')
            tk = tk.replace('enc_transformer.', 'transformer.')
            tk = tk.replace('dec_transformer.', 'transformer.')
            if tk.startswith('enc_'): tk = tk[4:]
            if tk.startswith('dec_'): tk = tk[4:]

            if tk in curr_sd:
                if v.shape == curr_sd[tk].shape:
                    new_state_dict[tk] = v
                elif 'emb_label.weight' in tk:
                    print(f"===> 正在移植 emb_label.weight (目標尺寸: {list(curr_sd[tk].shape)})")
                    new_v = curr_sd[tk].clone()
                    n_old = min(v.shape[0], new_v.shape[0])
                    new_v[:n_old] = v[:n_old]
                    new_state_dict[tk] = new_v

        netG.load_state_dict(new_state_dict, strict=False)
        num_matched = len(new_state_dict)
        print(f"===> 【最終載入結果】成功注入: {num_matched} 層權重")
        
        if num_matched < 50:
             print("警告：載入層數過低，請檢查 G_num_layers 是否設為 4 或 d_model 是否為 256！")
        else:
             print("成功：預訓練美感已成功注入模型。")
    else:
        print(f"===> 未找到預訓練模型，將從零開始訓練。")

    
    # === 固定樣本載入：建議使用含 pos 的 fixed_sample_v46.pt ===
    fixed_path = args.fixed_sample
    if not os.path.exists(fixed_path):
        print(f"致命錯誤：找不到 {fixed_path}。請先用 make_fixed_sample_v47.py 產生。")
        exit()

    ck = torch.load(fixed_path, weights_only=False, map_location=device)

    fixed_label = ck['label'].to(device).round().long()   # (B,N)
    fixed_mask  = ck['mask'].to(device).bool()            # (B,N) True=valid
    fixed_z     = ck['z'].to(device).float()              # (B,N,latent)

    fixed_pos = ck.get('pos', None)
    if fixed_pos is not None:
        fixed_pos = fixed_pos.to(device).float()

    fixed_face_pos = ck.get('face_pos', None)
    fixed_face_mask = ck.get('face_mask', None)
    if fixed_face_pos is not None:
        fixed_face_pos = fixed_face_pos.to(device).float()
    if fixed_face_mask is not None:
        fixed_face_mask = fixed_face_mask.to(device).bool()

    tmp = fixed_label[fixed_mask].reshape(-1)
    print("[fixed bincount]", torch.bincount(tmp, minlength=6).tolist())

    # 視覺化要畫 face：保留 fixed_mask 原樣即可（不再排除 FACE_ID）
    fixed_mask_noface = fixed_mask  # kept for backward compatibility; not used

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    start_iteration = load_training_checkpoint(
        args.resume_ckpt, device, netG, netD, optimizerG, optimizerD
    )

    stage_mult = get_stage_loss_multipliers(args.train_stage)
    print(f"[Stage] train_stage={args.train_stage} stage_mult={stage_mult}")

    torch.autograd.set_detect_anomaly(bool(args.detect_anomaly))

    iteration = start_iteration
    for epoch in range(10000):
        for data in train_dataloader:
            if iteration >= args.iteration: break
            label = data['x'].to(device)
            pos = data['pos'].to(device)
            mask = data['mask'].to(device)
            face_rel = data['face_rel'].to(device)
            face2img = data['face2img'].to(device)
            face_mask = data['face_mask'].to(device)
            bbox_real, padding_mask = pos, ~mask
            z = torch.randn(label.size(0), label.size(1), args.latent_size, device=device)

            # Update G
            netG.train(); netG.zero_grad()
            bbox_fake = torch.clamp(netG(z, label, padding_mask), 0, 1)
            # --- (2) 固定元素長寬比：只允許等比例縮放（避免元素被壓成細條） ---：只允許等比例縮放（避免元素被壓成細條） ---
            if args.fix_aspect:
                aspect_target_ids = (SVG_ID, IMG_ID) if not args.fix_text_aspect else (SVG_ID, TEXT_ID, IMG_ID)
                bbox_fake = project_fixed_aspect_scale(
                    bbox_fake, bbox_real, label, padding_mask,
                    target_ids=aspect_target_ids,
                )
            # --- (HARD) Face 與 Image 硬耦合：Face token 只輸出相對參數，Face bbox 由對應 Image 推導 ---
            # 注意：需要 face2img (dataset 產生) 來指定每張臉屬於哪張 image
            bbox_fake = hard_couple_faces_to_images(
                bbox_fake, label, padding_mask,
                face2img=face2img, face_mask=face_mask,
                face_rel_gt=face_rel, use_gt_face_rel=bool(args.freeze_face),
                keep_inside_image=True,
            )

            # --- (1) Image 固定 => Face 也固定：用 GT 直接覆蓋（避免 containment/face loss 逼到退化解） ---
            if args.fix_img_prob > 0:
                Bsz = label.size(0)
                fix_b = (torch.rand(Bsz, device=device) < args.fix_img_prob)
                if fix_b.any():
                    vmask = ~padding_mask
                    img_fix_m = (label == IMG_ID) & vmask & fix_b[:, None]
                    face_fix_m = (label == FACE_ID) & vmask & fix_b[:, None]
                    bbox_fake = torch.where(img_fix_m.unsqueeze(-1), bbox_real, bbox_fake)
                    bbox_fake = torch.where(face_fix_m.unsqueeze(-1), bbox_real, bbox_fake)


            # --- (3) freeze_face=1 時，不再把 face 絕對座標覆蓋成 GT。
            #     face 已在 hard_couple 中用「GT 相對座標 + 當前 image」重建，
            #     這樣 face 會永遠跟著 image，且不需要猜 face 自己的位置。

            # ---- Layout-only masking: D / align / overlap / style 都不看 Face/BG/Mask ----
            ignore_layout = (label == FACE_ID) | (label == BG_ID) | (label == MASK_ID)
            padding_layout = padding_mask | ignore_layout
            padding_overlap = padding_mask | (label == BG_ID) | (label == MASK_ID)
            layout_valid = ~padding_layout

            # 1. 基礎對抗損失（D 不看 face/bg/mask）
            loss_G_adv = F.softplus(-netD(bbox_fake, label, padding_layout)).mean()
            loss_G = loss_G_adv

            # --- [多樣化留白美學風格引導：用 margin-based style 分數(避免元素硬推到同一條線造成重疊)] ---
            valid_mask = layout_valid
            B = label.size(0)
            style_tags = make_style_tags(args, B, device)

            # style loss warmup：先讓 G 學會基本布局(不重疊/不崩壞)再開始強推風格
            curr_lambda_style = get_curr_lambda(args.lambda_style * stage_mult['style'], iteration, 5000, 5000)
            if curr_lambda_style > 0:
                loss_style = whitespace_style_loss(
                    bbox_fake,
                    label,
                    padding_layout,
                    wr_min=args.wr_min,
                    style_mode=args.ws_style_mode,
                    style_tags=style_tags,
                )
                loss_G += (loss_style * curr_lambda_style)

            curr_lambda_template = get_curr_lambda(
                args.lambda_template * stage_mult['template'], iteration, args.template_start, args.template_warmup
            )
            if curr_lambda_template > 0:
                loss_G += poster_template_loss(
                    bbox_fake,
                    label,
                    padding_layout,
                    template=args.layout_template,
                    lambda_template=curr_lambda_template,
                )

            curr_lambda_space = get_curr_lambda(args.lambda_space * stage_mult['space'], iteration, args.space_start, args.space_warmup)
            if curr_lambda_space > 0:
                loss_G += local_spacing_loss(
                    bbox_fake,
                    label,
                    padding_layout,
                    min_gap=args.space_min_gap,
                ) * curr_lambda_space

            # --- [對齊 loss：避免用 std 造成塌縮，改用 grid 吸附(只對非 face 元件)] ---
            curr_lambda_align = get_curr_lambda(args.lambda_align * stage_mult['align'], iteration, args.align_start, args.align_warmup)
            if curr_lambda_align > 0:
                if args.layout_template == 'center_vertical':
                    gx = [0.50]
                    gy = [0.08, 0.18, 0.30, 0.44, 0.58, 0.72, 0.86]
                    ew, cw = 0.5, 2.0
                elif args.layout_template == 'right_column':
                    gx = [0.64, 0.76, 0.88]
                    gy = [0.10, 0.22, 0.36, 0.52, 0.70, 0.86]
                    ew, cw = 1.0, 1.5
                elif args.layout_template == 'top_hero':
                    gx = [0.18, 0.50, 0.82]
                    gy = [0.10, 0.18, 0.42, 0.62, 0.82]
                    ew, cw = 0.8, 1.2
                elif args.layout_template == 'hybrid':
                    gx = [0.58, 0.74, 0.88]
                    gy = [0.22, 0.38, 0.56, 0.74, 0.90]
                    ew, cw = 0.9, 1.1
                else:  # bottom_caption
                    gx = [0.10, 0.50, 0.90]
                    gy = [0.12, 0.28, 0.48, 0.72, 0.84, 0.92]
                    ew, cw = 0.8, 1.2

                loss_G += grid_alignment_loss_xy(
                    bbox_fake,
                    label,
                    padding_layout,
                    lambda_grid=curr_lambda_align,
                    target_ids=(TEXT_ID, IMG_ID),
                    grid_x=gx,
                    grid_y=gy,
                    edge_weight=ew,
                    center_weight=cw,
                )
            # 4. 圖片幾何約束統一交給 image_geometry_regularizer()，
            #    避免與後面的 loss_img_geom 重複施壓。

            curr_lambda_text_width = get_curr_lambda(
                args.lambda_text_width * stage_mult['text_width'], iteration, args.text_width_start, args.text_width_warmup
            )
            curr_lambda_shape_prior = get_curr_lambda(
                args.lambda_shape_prior * stage_mult['shape_prior'], iteration, args.shape_prior_start, args.shape_prior_warmup
            )
            curr_lambda_role_layout = get_curr_lambda(
                args.lambda_role_layout * stage_mult['role_layout'], iteration, args.role_layout_start, args.role_layout_warmup
            )
            curr_lambda_region = get_curr_lambda(
                args.lambda_region * stage_mult['region'], iteration, args.region_start, args.region_warmup
            )

            loss_text_width = torch.tensor(0.0, device=device)
            loss_text_geom = torch.tensor(0.0, device=device)
            loss_img_geom = torch.tensor(0.0, device=device)
            loss_shape_prior = torch.tensor(0.0, device=device)
            loss_role_layout = torch.tensor(0.0, device=device)
            loss_region = torch.tensor(0.0, device=device)
            max_text_w = args.text_max_w_top_hero if args.layout_template == 'top_hero' else args.text_max_w
            if curr_lambda_text_width > 0:
                loss_text_width = text_width_regularizer(
                    bbox_fake,
                    label,
                    padding_layout,
                    text_id=TEXT_ID,
                    min_w=args.text_min_w,
                    max_w=max_text_w,
                    edge_margin=args.text_edge_margin,
                    too_narrow_weight=args.text_too_narrow_weight,
                    too_wide_weight=args.text_too_wide_weight,
                )
                loss_G += curr_lambda_text_width * loss_text_width

            curr_lambda_text_geom = args.lambda_text_geom * stage_mult['text_geom']
            if curr_lambda_text_geom > 0:
                loss_text_geom = text_geometry_regularizer(
                    bbox_fake,
                    label,
                    padding_layout,
                    text_id=TEXT_ID,
                    min_h=args.text_min_h,
                    min_area=args.text_min_area,
                    max_aspect=args.text_max_aspect,
                )
                loss_G += curr_lambda_text_geom * loss_text_geom

            curr_lambda_img_geom = args.lambda_img_geom * stage_mult['img_geom']
            if curr_lambda_img_geom > 0:
                loss_img_geom = image_geometry_regularizer(
                    bbox_fake,
                    label,
                    padding_layout,
                    img_id=IMG_ID,
                    min_area=args.img_min_area,
                    min_short_side=args.img_min_short_side,
                    min_aspect=args.img_min_aspect,
                    max_aspect=args.img_max_aspect,
                )
                loss_G += curr_lambda_img_geom * loss_img_geom

            if curr_lambda_shape_prior > 0:
                loss_shape_prior = batch_shape_prior_loss(
                    bbox_fake,
                    bbox_real,
                    label,
                    padding_mask,
                    target_ids=(TEXT_ID, IMG_ID, SVG_ID),
                )
                loss_G += curr_lambda_shape_prior * loss_shape_prior

            if args.layout_template == 'center_vertical' and curr_lambda_role_layout > 0:
                loss_role_layout = center_vertical_role_loss(
                    bbox_fake,
                    label,
                    padding_layout,
                )
                loss_G += curr_lambda_role_layout * loss_role_layout

            if curr_lambda_region > 0:
                loss_region = content_region_loss(
                    bbox_fake,
                    label,
                    padding_layout,
                    template=args.layout_template,
                    lambda_region=curr_lambda_region,
                    center_weight=args.region_center_weight,
                    compact_weight=args.region_compact_weight,
                )
                loss_G += loss_region

            # 5. [其餘幾何損失加總]
            fake_wh = bbox_fake[valid_mask][:, 2:]
            aspect_ratio = fake_wh[:, 0] / (fake_wh[:, 1] + 1e-6)
            loss_aspect = (torch.relu(aspect_ratio - 8.0) + torch.relu(0.125 - aspect_ratio)).mean()
            loss_min_size = torch.relu(0.02 - (fake_wh[:, 0] * fake_wh[:, 1])).mean()
            curr_lambda_ov = min(args.lambda_ov, args.lambda_ov * (iteration / 5000.0))

            curr_lambda_face = get_curr_lambda(args.lambda_face * stage_mult['face'], iteration, args.face_start, args.face_warmup)
            curr_lambda_cont = get_curr_lambda(args.lambda_cont * stage_mult['cont'], iteration, args.cont_start, args.cont_warmup)
            curr_lambda_anchor = get_curr_lambda(args.lambda_anchor * stage_mult['anchor'], iteration, args.anchor_start, args.anchor_warmup)

            loss_G += (curr_lambda_ov * pairwise_overlap_loss(bbox_fake, label, padding_overlap)) + \
                      face_coverage_loss(bbox_fake, label, padding_mask, curr_lambda_face) + \
                      element_size_loss(bbox_fake, label, padding_mask, [IMG_ID], 0.35) + \
                      containment_loss(bbox_fake, label, padding_mask, lambda_cont=curr_lambda_cont) + \
                      face_anchor_loss_relative(bbox_fake, label, padding_mask, face_rel, face_mask, face2img, lambda_anchor=curr_lambda_anchor) + \
                      (loss_aspect * 100.0) + (loss_min_size * 100.0)

            loss_G.backward()
            optimizerG.step()

            # Update D
            netD.zero_grad()
            loss_D = F.softplus(netD(bbox_fake.detach(), label, padding_layout)).mean() + \
                     F.softplus(-netD(bbox_real, label, padding_layout)).mean()
            loss_D.backward(); optimizerD.step()

            if iteration % 100 == 0:
                vmask = ~padding_mask
                n_face = int(((label == FACE_ID) & vmask).sum().item())
                n_occ  = int((((label == TEXT_ID) | (label == SVG_ID) | (label == MASK_ID)) & vmask).sum().item())
                print("[debug unique labels in batch]", torch.unique(label[~padding_mask]).tolist())
                print(f"[occ debug] iter={iteration} n_face_tokens={n_face} n_occluder_tokens={n_occ}")

                with torch.no_grad():
                    occ_ratio, occ_rate = face_occlusion_stats(bbox_fake, label, padding_mask, thr=0.01)
                    occ_ratio_mean = occ_ratio
                    union_w_mean, union_h_mean, union_x1_mean, union_y1_mean, union_count_mean, union_area_sum_mean = content_union_stats(
                        bbox_fake, label, padding_layout
                    )

                with open(occ_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([iteration, float(occ_ratio_mean.item()), float(occ_rate.item()), n_face, n_occ])

                print(f'[{iteration}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} '
                      f'STYLE:{curr_lambda_style:.3f} SPACE:{curr_lambda_space:.3f} REGION:{loss_region.item():.3f} TXTW:{(curr_lambda_text_width * loss_text_width).item():.3f} '
                      f'TGEO:{(curr_lambda_text_geom * loss_text_geom).item():.3f} IGEO:{(curr_lambda_img_geom * loss_img_geom).item():.3f} '
                      f'SHAPE:{(curr_lambda_shape_prior * loss_shape_prior).item():.3f} ROLE:{(curr_lambda_role_layout * loss_role_layout).item():.3f} '
                      f'OV:{curr_lambda_ov:.3f} ALIGN:{curr_lambda_align:.3f} '
                      f'FACE:{curr_lambda_face:.3f} CONT:{curr_lambda_cont:.1f} ANCH:{curr_lambda_anchor:.1f} '
                      f'UW:{union_w_mean.item():.3f} UH:{union_h_mean.item():.3f} UX1:{union_x1_mean.item():.3f} UY1:{union_y1_mean.item():.3f} '
                      f'UC:{union_count_mean.item():.2f} UA:{union_area_sum_mean.item():.3f} '
                      f'OCC:{occ_ratio.item():.3f} OCC_RATE:{occ_rate.item():.2f}')

            if args.save_every > 0 and iteration > 0 and iteration % args.save_every == 0:
                save_training_checkpoint(
                    ckpt_dir, iteration, netG, netD, optimizerG, optimizerD, args
                )

            if iteration % args.vis_every == 0:
                netG.eval()
                with torch.no_grad():
                    fixed_padding = ~fixed_mask  # True = padding
                    bbox_vis = torch.clamp(netG(fixed_z, fixed_label, fixed_padding), 0, 1)

                    # 若 fixed_sample 有 pos，套用與訓練一致的 post-process（讓預覽更可信）
                    if fixed_pos is not None:
                        if args.fix_aspect:
                            aspect_target_ids = (SVG_ID, IMG_ID) if not args.fix_text_aspect else (SVG_ID, TEXT_ID, IMG_ID)
                            bbox_vis = project_fixed_aspect_scale(
                                bbox_vis, fixed_pos, fixed_label, fixed_padding,
                                target_ids=aspect_target_ids,
                            )

                        # --- (HARD) VIS: Face 與 Image 硬耦合 (mapping 從 GT pos 推導) ---
                        fixed_face2img = infer_face2img_from_reference(
                            fixed_pos, fixed_label, fixed_padding,
                            max_faces=args.max_faces, contain_thr=0.98,
                        )
                        # fixed_face_mask：用 label/valid 推估（最多 4 張）
                        fixed_face_mask_vis = torch.zeros((fixed_label.size(0), args.max_faces), device=device, dtype=torch.bool)
                        vmask = ~fixed_padding
                        for bb in range(fixed_label.size(0)):
                            f_idx = torch.where((fixed_label[bb] == FACE_ID) & vmask[bb])[0]
                            nf = min(int(f_idx.numel()), args.max_faces)
                            if nf > 0:
                                fixed_face_mask_vis[bb, :nf] = True
                        fixed_face_rel_vis = build_face_rel_gt_from_reference(
                            fixed_pos, fixed_label, fixed_padding, fixed_face2img,
                            max_faces=args.max_faces,
                        )
                        bbox_vis = hard_couple_faces_to_images(
                            bbox_vis, fixed_label, fixed_padding,
                            face2img=fixed_face2img, face_mask=fixed_face_mask_vis,
                            face_rel_gt=fixed_face_rel_vis, use_gt_face_rel=bool(args.freeze_face),
                            keep_inside_image=True,
                        )
                    else:
                        if args.fix_aspect or args.freeze_face:
                            print("[WARN] fixed_sample 沒有 'pos'，預覽圖是 raw netG output（不含 fix_aspect / freeze_face）。")

                    # 視覺化：保留 face（但仍排除 BG/MASK）
                    final_vis_mask = fixed_mask & (fixed_label != BG_ID) & (fixed_label != MASK_ID)

                    vis_save_path = os.path.join(out_dir, f'fake_{iteration:05d}.png')
                    colors_pil = [tuple(int(x) for x in c) for c in train_dataset.colors]
                    final_vis_mask = final_vis_mask & (fixed_label < len(colors_pil))

                    save_image(bbox_vis, fixed_label, final_vis_mask, colors_pil, vis_save_path)
                    print(f"===> 已儲存預覽圖 (含 face): {vis_save_path}")
                netG.train()
            iteration += 1


    # 保證結束時一定存 final checkpoint / final preview，避免剛好停在 iteration 上限卻沒進到上面的 save/vis 區塊
    final_iter = min(iteration, args.iteration)

    if args.save_every > 0:
        save_training_checkpoint(
            ckpt_dir, final_iter, netG, netD, optimizerG, optimizerD, args
        )

    netG.eval()
    with torch.no_grad():
        fixed_padding = ~fixed_mask  # True = padding
        bbox_vis = torch.clamp(netG(fixed_z, fixed_label, fixed_padding), 0, 1)

        # 若 fixed_sample 有 pos，套用與訓練一致的 post-process（讓預覽更可信）
        if fixed_pos is not None:
            if args.fix_aspect:
                aspect_target_ids = (SVG_ID, IMG_ID) if not args.fix_text_aspect else (SVG_ID, TEXT_ID, IMG_ID)
                bbox_vis = project_fixed_aspect_scale(
                    bbox_vis, fixed_pos, fixed_label, fixed_padding,
                    target_ids=aspect_target_ids,
                )

            fixed_face2img = infer_face2img_from_reference(
                fixed_pos, fixed_label, fixed_padding,
                max_faces=args.max_faces, contain_thr=0.98,
            )
            fixed_face_mask_vis = torch.zeros((fixed_label.size(0), args.max_faces), device=device, dtype=torch.bool)
            vmask = ~fixed_padding
            for bb in range(fixed_label.size(0)):
                f_idx = torch.where((fixed_label[bb] == FACE_ID) & vmask[bb])[0]
                nf = min(int(f_idx.numel()), args.max_faces)
                if nf > 0:
                    fixed_face_mask_vis[bb, :nf] = True
            fixed_face_rel_vis = build_face_rel_gt_from_reference(
                fixed_pos, fixed_label, fixed_padding, fixed_face2img,
                max_faces=args.max_faces,
            )
            bbox_vis = hard_couple_faces_to_images(
                bbox_vis, fixed_label, fixed_padding,
                face2img=fixed_face2img, face_mask=fixed_face_mask_vis,
                face_rel_gt=fixed_face_rel_vis, use_gt_face_rel=bool(args.freeze_face),
                keep_inside_image=True,
            )
        else:
            if args.fix_aspect or args.freeze_face:
                print("[WARN] fixed_sample 沒有 'pos'，最終預覽圖是 raw netG output（不含 fix_aspect / freeze_face）。")

        final_vis_mask = fixed_mask & (fixed_label != BG_ID) & (fixed_label != MASK_ID)
        colors_pil = [tuple(int(x) for x in c) for c in train_dataset.colors]
        final_vis_mask = final_vis_mask & (fixed_label < len(colors_pil))

        final_vis_path = os.path.join(out_dir, f'fake_{final_iter:05d}.png')
        save_image(bbox_vis, fixed_label, final_vis_mask, colors_pil, final_vis_path)
        print(f"===> 已儲存最終預覽圖 (含 face): {final_vis_path}")
    netG.train()
if __name__ == "__main__":
    main()