#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
產生固定樣本 fixed_sample_v46.pt（包含 pos），用於 LayoutGAN++ 訓練時固定可視化。

輸出內容（torch.save dict）：
- label: (B, N) long
- pos:   (B, N, 4) float
- mask:  (B, N) bool     # True=有效 token
- face_pos: (B, max_faces, 4) float  (若 dataset 提供)
- face_mask:(B, max_faces) bool     (若 dataset 提供)
- z:     (B, N, latent_size) float  # 固定 latent

用法：
python make_fixed_sample.py --out fixed_sample_v46.pt --bs 64 --latent_size 4
"""

import os
import argparse
import pickle
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from data import get_dataset

# ========= Constants =========
SVG_ID = 0
TEXT_ID = 1
IMG_ID = 2
BG_ID = 3
MASK_ID = 4
FACE_ID = 5


class RawLayoutDataset(torch.utils.data.Dataset):
    """
    與 train.py 一致的封裝：face 不吃 max_nodes 配額，最後再把 face 合併回 token 序列，
    同時回傳 face_pos/face_mask 供 anchor loss 使用。
    """
    def __init__(self, data_list, num_classes, max_nodes=4, max_faces=4, colors=None):
        self.data_list = data_list
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.max_faces = max_faces
        self.colors = colors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        bbox, label = self.data_list[idx]
        bbox = np.array(bbox)
        label = np.array(label)

        # 1) split face / others
        face_mask_idx = (label == FACE_ID)
        temp_f_label = label[face_mask_idx]
        temp_f_bbox = bbox[face_mask_idx]
        o_label = label[~face_mask_idx]
        o_bbox = bbox[~face_mask_idx]

        # 2) 非 face 元件：把大面積 SVG 轉成 IMG，然後按優先順序 + 面積取前 max_nodes
        if len(o_label) > 0:
            area = o_bbox[:, 2] * o_bbox[:, 3]
            o_label[(o_label == SVG_ID) & (area > 0.15)] = IMG_ID

        n = len(o_label)
        if n > self.max_nodes:
            area = o_bbox[:, 2] * o_bbox[:, 3]
            idxs = np.arange(n)
            priority = np.ones(n, dtype=np.int64)
            priority[o_label == IMG_ID] = 0
            priority[o_label == TEXT_ID] = 1
            priority[o_label == SVG_ID] = 2
            priority[o_label == BG_ID] = 3
            priority[o_label == MASK_ID] = 4
            order = np.lexsort((idxs, -area, priority))
            keep = np.sort(order[:self.max_nodes])
            o_label = o_label[keep]
            o_bbox = o_bbox[keep]

        # 3) face：直接取前 max_faces（保留原始順序）
        f_label = temp_f_label
        f_bbox = temp_f_bbox
        if len(f_label) == 0:
            f_label = np.array([], dtype=np.int64)
            f_bbox = np.empty((0, 4), dtype=np.float32)

        face_pos = torch.zeros((self.max_faces, 4), dtype=torch.float)
        face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
        nf = min(len(f_label), self.max_faces)
        if nf > 0:
            face_pos[:nf] = torch.FloatTensor(f_bbox[:nf])
            face_mask[:nf] = True

        # 4) merge + pad 到 total_cap
        final_label = np.concatenate([o_label, f_label])
        final_bbox = np.concatenate([o_bbox, f_bbox])

        total_cap = self.max_nodes + self.max_faces
        pad_x = torch.full((total_cap,), self.num_classes - 1, dtype=torch.long)
        pad_pos = torch.zeros((total_cap, 4), dtype=torch.float)
        pad_mask = torch.zeros((total_cap,), dtype=torch.bool)

        curr_total = len(final_label)
        if curr_total > 0:
            pad_x[:curr_total] = torch.LongTensor(final_label)
            pad_pos[:curr_total] = torch.FloatTensor(final_bbox)
            pad_mask[:curr_total] = True

        return {
            'x': pad_x,
            'pos': pad_pos,
            'mask': pad_mask,
            'face_pos': face_pos,
            'face_mask': face_mask,
        }


def build_dataset(args):
    base = get_dataset(args.dataset, 'train')
    num_classes = base.num_classes

    # optional: pkl override
    ds = base
    if args.pkl_path and os.path.exists(args.pkl_path):
        with open(args.pkl_path, 'rb') as f:
            high_quality_data = pickle.load(f)
        ds = RawLayoutDataset(high_quality_data, num_classes=num_classes, colors=base.colors)

    # colors（保留 train.py 的固定色表，避免 TypeError）
    ds.colors = [
        (31, 119, 180),   # 0: SVG
        (44, 160, 44),    # 1: TEXT
        (148, 103, 189),  # 2: IMG
        (227, 119, 194),  # 3: BG
        (188, 189, 34),   # 4: MASK
        (158, 218, 229),  # 5: FACE
    ]
    ds.colors = [tuple(int(c) for c in color) for color in ds.colors]
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='crello')
    parser.add_argument('--pkl_path', type=str,
                        default="/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_ws_gt0.6_with_face.pkl")
    parser.add_argument('--out', type=str, default='fixed_sample_v46.pt')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ds = build_dataset(args)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    batch = next(iter(loader))

    label = batch['x'].round().long().cpu()
    pos   = batch['pos'].float().cpu()
    mask  = batch['mask'].bool().cpu()
    face_pos  = batch.get('face_pos', None)
    face_mask = batch.get('face_mask', None)
    if face_pos is not None:  face_pos = face_pos.float().cpu()
    if face_mask is not None: face_mask = face_mask.bool().cpu()

    B, N = label.shape
    z = torch.randn(B, N, args.latent_size).float().cpu()

    payload = {'label': label, 'pos': pos, 'mask': mask, 'z': z}
    if face_pos is not None:  payload['face_pos']  = face_pos
    if face_mask is not None: payload['face_mask'] = face_mask

    torch.save(payload, args.out)
    print(f"===> saved: {args.out}")
    print(f"     label: {tuple(label.shape)}  pos: {tuple(pos.shape)}  mask: {tuple(mask.shape)}  z: {tuple(z.shape)}")
    if face_pos is not None:
        print(f"     face_pos: {tuple(face_pos.shape)}  face_mask: {tuple(face_mask.shape)}")


if __name__ == '__main__':
    main()
