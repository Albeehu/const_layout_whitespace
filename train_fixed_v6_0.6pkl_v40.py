"""用 train_fixed_v6_0.6pkl_v38 改 v40
    1. 用fixed_sample_v45 因為 v18改壞了
    2. /home/albee/const_layout_whitespace/data/dataset/crello/crello_train_ws_gt0.5_with_face.pkl
       換成pkl_path = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_ws_gt0.6_with_face.pkl"
"""
import os
import csv
import argparse
import pickle
from pathlib import Path
import numpy as np
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

class RawLayoutDataset(torch.utils.data.Dataset):
    """
    更新邏輯：
    1. Face (ID=5) 不計入 max_nodes=4 的配額。
    2. 其他元件依優先權（Image > Text > SVG > BG > Mask）保留最多 6 個。
    3. 移除 Face 面積排序，直接按原始順序取前 max_faces 個。
    4. 最後將 Face 合併回生成清單，確保生成器會輸出人臉框。
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

            # 1. 分離人臉與其他元件
            face_mask_idx = (label == FACE_ID)
            temp_f_label = label[face_mask_idx]
            temp_f_bbox = bbox[face_mask_idx]
            o_label = label[~face_mask_idx]
            o_bbox = bbox[~face_mask_idx]

            # 2. 處理非人臉元件 (max_nodes=6)
            if len(o_label) > 0:
                area = o_bbox[:, 2] * o_bbox[:, 3]
                o_label[(o_label == 0) & (area > 0.15)] = IMG_ID

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

            # 3. [修正版] 直接保留所有人臉，不進行 is_contained 過濾
            # 這樣生成器才能學到人臉與圖片的對應關係
            f_label = temp_f_label
            f_bbox = temp_f_bbox

            # 如果該樣本完全沒有臉，仍需處理空陣列
            if len(f_label) == 0:
                f_label = np.array([], dtype=np.int64)
                f_bbox = np.empty((0, 4))
            
            # 4. 初始化回傳給 Face Anchor Loss 的數據
            # 只要原始數據有臉，face_mask 就是 True，強制模型去學習對齊
            face_pos = torch.zeros((self.max_faces, 4), dtype=torch.float)
            face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
            nf = min(len(f_label), self.max_faces)
            if nf > 0:
                face_pos[:nf] = torch.FloatTensor(f_bbox[:nf])
                face_mask[:nf] = True

            # 4. 合併與 Padding
            final_label = np.concatenate([o_label, f_label])
            final_bbox = np.concatenate([o_bbox, f_bbox])
            
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
                'face_pos': face_pos,
                'face_mask': face_mask,
            }



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
    bbox_pred, bbox_ref, label, padding_mask,
    target_ids=(2, 3, 4),   # 你自己對應：SVG/TEXT/IMG 的 id
    eps=1e-6,
):
    """
    bbox_pred, bbox_ref: (B, N, 4) in (cx, cy, w, h) normalized to [0,1]
    Return: same shape, but for target_ids keep aspect ratio from bbox_ref and only scale uniformly.
    全程 non-inplace，避免 autograd 版本號錯誤。
    """
    B, N, _ = bbox_pred.shape
    device = bbox_pred.device

    valid = ~padding_mask  # (B,N) bool
    tgt = torch.zeros((B, N), device=device, dtype=torch.bool)
    for tid in target_ids:
        tgt = tgt | ((label == tid) & valid)   # 非 inplace（不要用 |=）

    # clamp to avoid divide-by-zero
    w0 = bbox_ref[..., 2].clamp_min(eps)
    h0 = bbox_ref[..., 3].clamp_min(eps)
    w  = bbox_pred[..., 2].clamp_min(eps)
    h  = bbox_pred[..., 3].clamp_min(eps)

    # keep area of pred, but force aspect ratio of ref
    s = torch.sqrt((w * h) / (w0 * h0))
    w_new = w0 * s
    h_new = h0 * s

    cx = bbox_pred[..., 0]
    cy = bbox_pred[..., 1]

    new_bbox = torch.stack([cx, cy, w_new, h_new], dim=-1)  # (B,N,4)

    # only replace target elements
    bbox_out = torch.where(tgt.unsqueeze(-1), new_bbox, bbox_pred)
    return bbox_out

    w0 = bbox_ref[:, :, 2].clamp(min=eps)
    h0 = bbox_ref[:, :, 3].clamp(min=eps)
    w = bbox_pred[:, :, 2].clamp(min=eps)
    h = bbox_pred[:, :, 3].clamp(min=eps)

    s = torch.sqrt((w * h) / (w0 * h0)).clamp(min=0.25, max=4.0)
    w_new = (w0 * s).clamp(min=eps, max=1.0)
    h_new = (h0 * s).clamp(min=eps, max=1.0)

    out = bbox_pred.clone()
    out[:, :, 2] = torch.where(tgt, w_new, out[:, :, 2])
    out[:, :, 3] = torch.where(tgt, h_new, out[:, :, 3])

    # 讓中心點不超出邊界
    half_w = out[:, :, 2] / 2.0
    half_h = out[:, :, 3] / 2.0
    cx = out[:, :, 0].clamp(half_w, 1.0 - half_w)
    cy = out[:, :, 1].clamp(half_h, 1.0 - half_h)
    out[:, :, 0] = cx
    out[:, :, 1] = cy

    return out
def whitespace_style_loss(bbox_fake, padding_mask, wr_min=0.6, w_style=1.0, style_mode="max", style_tags=None):
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


def alignment_loss(bbox_fake, padding_mask, lambda_align=5.0):
    device = bbox_fake.device
    valid_boxes = bbox_fake[~padding_mask]
    if valid_boxes.size(0) == 0: return torch.tensor(0.0, device=device)
    c_pts = valid_boxes[:, :2]; dist = torch.abs(c_pts - 0.5)
    return lambda_align * dist.mean()

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

def zero_diag(mat: torch.Tensor) -> torch.Tensor:
    """Return mat with diagonal set to 0, WITHOUT inplace ops. mat shape (..., N, N)."""
    n = mat.size(-1)
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    while eye.dim() < mat.dim():
        eye = eye.unsqueeze(0)
    return mat * (1.0 - eye)

def add_to_diag(mat: torch.Tensor, val: float) -> torch.Tensor:
    """Return mat with diagonal += val, WITHOUT inplace ops. mat shape (..., N, N)."""
    n = mat.size(-1)
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    while eye.dim() < mat.dim():
        eye = eye.unsqueeze(0)
    return mat + eye * val

def pairwise_overlap_loss(bbox_fake, label, padding_mask):
    device = bbox_fake.device
    W = torch.ones((10, 10), device=device)
    W[BG_ID, :], W[:, BG_ID], W[MASK_ID, :], W[:, MASK_ID] = 0.0, 0.0, 0.0, 0.0
    W[IMG_ID, FACE_ID], W[FACE_ID, IMG_ID] = 0.0, 0.0 
    W[1, 1], W[1, FACE_ID], W[FACE_ID, 1] = 10.0, 25.0, 25.0  # 避免雙重懲罰：face_coverage_loss 已經在管文字壓臉
    
    losses = []
    for b in range(bbox_fake.size(0)):
        valid = ~padding_mask[b]
        if valid.sum() <= 1: continue
        xyxy = xywh_to_xyxy(bbox_fake[b][valid]); labs = label[b][valid]
        inter = box_intersection_area_xyxy(xyxy, xyxy)
        n = inter.size(0)
        eye = torch.eye(n, device=inter.device, dtype=inter.dtype)
        inter = inter * (1.0 - eye)   # 不用 fill_diagonal_，避免 inplace
        area = box_area_xyxy(xyxy, eps=1e-4)
        denom = torch.minimum(area[:, None], area[None, :]).clamp(min=1e-4)
        overlap_ratio = (inter / denom).clamp(max=2.0)
        loss_b = (overlap_ratio * W[labs[:, None], labs[None, :]]).triu(1).sum()
        losses.append(loss_b / ((valid.sum() * (valid.sum() - 1)) / 2).clamp(min=1))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

def face_anchor_loss(bbox_fake, label, padding_mask, face_pos, face_mask, lambda_anchor=50.0):
    """
    強迫生成器預測的人臉框 (bbox_fake[label==5]) 去對齊真實的人臉座標 (face_pos)。
    這能教導生成器「臉通常長在哪裡」。
    """
    device = bbox_fake.device
    B = bbox_fake.size(0)
    loss, count = torch.tensor(0.0, device=device), 0
    
    for b in range(B):
        f_idx_in_gen = torch.where((label[b] == FACE_ID) & (~padding_mask[b]))[0]
        num_gt_faces = face_mask[b].sum()
        
        n = min(len(f_idx_in_gen), num_gt_faces)
        if n > 0:
            # 假設順序一致（因為 RawLayoutDataset 的拼接順序是固定的）
            loss += F.mse_loss(bbox_fake[b][f_idx_in_gen[:n]], face_pos[b][:n])
            count += 1
            
    return lambda_anchor * (loss / count) if count > 0 else loss



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='/home/albee/const_layout_whitespace/pretrained/layoutnet_crello.pth.tar')
    parser.add_argument('--name', type=str, default='v13_ft')
    parser.add_argument('--occ_thresh', type=float, default=0.01)
    parser.add_argument('--dataset', type=str, default='crello')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iteration', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--lambda_ws', type=float, default=0.08)
    parser.add_argument('--wr_min', type=float, default=0.6)
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
    parser.add_argument('--fix_aspect', type=int, default=1)           # 1=固定長寬比(只等比縮放), 0=不啟用
    parser.add_argument('--ws_style_mode', type=str, default='random', choices=['max','random'])  # 留白風格選擇方式
    parser.add_argument('--fix_img_prob', type=float, default=0.0)     # 以機率將 Image(以及 Face) 固定到 GT (0~1)
    parser.add_argument('--freeze_face', type=int, default=1)        # 1=Face永遠用GT(不讓模型猜), 0=關閉
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--aug_flip', action='store_true')
    parser.add_argument('--vis_every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fixed_sample', type=str, default='fixed_sample_v45.pt') # 確保檔名正確
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--G_d_model', type=int, default=256)
    parser.add_argument('--G_nhead', type=int, default=4)
    parser.add_argument('--G_num_layers', type=int, default=4)
    parser.add_argument('--D_d_model', type=int, default=256)
    parser.add_argument('--D_nhead', type=int, default=4)
    parser.add_argument('--D_num_layers', type=int, default=4)
    args = parser.parse_args()

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
    # 1. 獲取原始 dataset
    train_dataset = get_dataset(args.dataset, 'train')
    
    # 2. 如果有生成的 pkl 資料，重新封裝
    pkl_path = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_ws_gt0.6_with_face.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f: 
            high_quality_data = pickle.load(f)
        train_dataset = RawLayoutDataset(high_quality_data, num_classes=train_dataset.num_classes, colors=train_dataset.colors)
    
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
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # --- [關鍵修正 B：確保 dataloader 一定會被定義] ---
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
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

    # === 固定樣本載入 (v18 專用) ===
    fixed_path = 'fixed_sample_v45.pt'
    if not os.path.exists(fixed_path):
        print(f"致命錯誤：找不到 {fixed_path}。請先執行產生腳本產出二進位檔案。")
        exit()

    # 確保使用 map_location 以相容不同設備
    ck = torch.load(fixed_path, weights_only=False, map_location=device)
    fixed_label = ck['label'].to(device).round().long()   #  轉成整數 label
    fixed_mask  = ck['mask'].to(device).bool()            # mask 轉 bool
    fixed_z     = ck['z'].to(device).float()

    tmp = fixed_label[fixed_mask].reshape(-1)   # 用 tmp，不要回寫 fixed_label
    print("[fixed bincount]", torch.bincount(tmp, minlength=6).tolist())
    # Face 不作為 layout node：在視覺化時過濾掉 ID 5
    fixed_mask_noface = fixed_mask & (fixed_label != FACE_ID)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    
    torch.autograd.set_detect_anomaly(True)

    iteration = 0
    for epoch in range(10000):
        for data in train_dataloader:
            if iteration >= args.iteration: break
            label = data['x'].to(device)
            pos = data['pos'].to(device)
            mask = data['mask'].to(device)
            face_pos = data['face_pos'].to(device)
            face_mask = data['face_mask'].to(device)
            bbox_real, padding_mask = pos, ~mask
            z = torch.randn(label.size(0), label.size(1), args.latent_size, device=device)

            # Update G
            netG.train(); netG.zero_grad()
            bbox_fake = torch.clamp(netG(z, label, padding_mask), 0, 1)
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

            # --- (2) 固定元素長寬比：只允許等比例縮放（避免元素被壓成細條） ---
            if args.fix_aspect:
                bbox_fake = project_fixed_aspect_scale(
                    bbox_fake, bbox_real, label, padding_mask,
                    target_ids=(SVG_ID, TEXT_ID, IMG_ID),
                )

            
                        # --- (3) Face 永遠固定到 GT：不讓模型猜 Face 位置 ---
            if args.freeze_face:
                vmask = ~padding_mask
                face_m = (label == FACE_ID) & vmask
                if face_m.any():
                    bbox_fake = torch.where(face_m.unsqueeze(-1), bbox_real, bbox_fake)
# 1. 基礎對抗損失
            loss_G_adv = F.softplus(-netD(bbox_fake, label, padding_mask)).mean()
            loss_G = loss_G_adv

            # --- [多樣化留白美學風格引導] ---
            valid_mask = ~padding_mask
            # 只讓「排版元件」參與風格/對齊：排除 Face / BG / Mask
            layout_mask = mask & (label != FACE_ID) & (label != BG_ID) & (label != MASK_ID)
            cx = bbox_fake[:, :, 0]
            cy = bbox_fake[:, :, 1]
            # --- [多樣化留白美學：風格選擇邏輯優化] ---
            # 建議：利用 iteration 來控制風格區間，例如每 1000 步換一種主打風格
            # 或是讓 batch 內分組進行，不要讓整個 batch 的目標雜亂
            # 讓同一個 Batch 內同時出現 4 種風格，Discriminator 才會學會接受多樣性
            B = label.size(0)
            style_tags = torch.randint(0, 4, (B,), device=device) if args.ws_style_mode == 'random' else (torch.arange(B, device=device) % 4)
            
            # --- [計算多樣化留白風格與一致性損失] ---
            valid_mask = ~padding_mask
            
            # [修正點]：先定義座標變數，才能給後面的 Loss 使用
            cx_f = bbox_fake[:, :, 0] # 中心 X
            cy_f = bbox_fake[:, :, 1] # 中心 Y
            # 計算左邊界 x1 = cx - w/2
            x1_f = (cx_f - bbox_fake[:, :, 2] / 2).clamp(0, 1)
            # 計算右邊界 x2 = cx + w/2
            x2_f = (cx_f + bbox_fake[:, :, 2] / 2).clamp(0, 1)
            # 計算頂邊 y1 = cy - h/2
            y1_f = (cy_f - bbox_fake[:, :, 3] / 2).clamp(0, 1)
            # 計算底邊 y2 = cy + h/2
            y2_f = (cy_f + bbox_fake[:, :, 3] / 2).clamp(0, 1)

            # 隨機或循環選擇風格 (例如 style_idx = (iteration // 2000) % 4)
            style_idx = (iteration // 2000) % 4 
            loss_style = torch.tensor(0.0, device=device)

            if layout_mask.any():
                if style_idx == 0: # Style A: 右靠 (左側留白)
                    loss_style = torch.abs(x2_f[layout_mask] - 0.85).mean()
                elif style_idx == 1: # Style B: 置中
                    loss_style = torch.abs(cx_f[layout_mask] - 0.5).mean()
                elif style_idx == 2: # Style C: 上下
                    img_m = (label == IMG_ID) & layout_mask
                    txt_m = (label == TEXT_ID) & layout_mask
                    if img_m.any() and txt_m.any():
                        loss_style = torch.abs(y1_f[img_m] - 0.2).mean() + torch.abs(y2_f[txt_m] - 0.8).mean()
                elif style_idx == 3: # Style D: 角落
                    loss_style = torch.abs(x2_f[layout_mask] - 0.95).mean() + torch.abs(y2_f[layout_mask] - 0.95).mean()

            curr_lambda_style = args.lambda_style * min(1.0, max(0.0, (iteration - 2000) / 3000.0))
            loss_G += (loss_style * curr_lambda_style)
            # --- [對齊 loss：避免用 std 造成塌縮，改用 grid 吸附(只對非 face 元件)] ---
            curr_lambda_align = args.lambda_align * min(1.0, max(0.0, (iteration - args.align_start) / max(1.0, float(args.align_warmup))))
            if curr_lambda_align > 0:
                loss_G += grid_alignment_loss_xy(
                    bbox_fake,
                    label,
                    padding_mask,
                    lambda_grid=curr_lambda_align,
                    target_ids=(SVG_ID, TEXT_ID, IMG_ID),
                )
            # 4. [圖片面積與形狀保護]
            img_mask = (label == IMG_ID) & valid_mask
            if img_mask.any():
                # 圖片面積至少 10%，防止為了留白而縮小
                img_areas = bbox_fake[img_mask][:, 2] * bbox_fake[img_mask][:, 3]
                loss_G += torch.relu(0.10 - img_areas).mean() * 150.0

            text_mask = (label == TEXT_ID) & valid_mask
            if text_mask.any():
                # 文字寬度保護，防止變成細線
                loss_text_width = torch.relu(0.25 - bbox_fake[text_mask][:, 2]).mean()
                loss_G += (loss_text_width * 100.0)

            # 5. [其餘幾何損失加總]
            fake_wh = bbox_fake[valid_mask][:, 2:]
            aspect_ratio = fake_wh[:, 0] / (fake_wh[:, 1] + 1e-6)
            loss_aspect = (torch.relu(aspect_ratio - 8.0) + torch.relu(0.125 - aspect_ratio)).mean()
            loss_min_size = torch.relu(0.05 - (fake_wh[:, 0] * fake_wh[:, 1])).mean()
            
            curr_lambda_ws = args.lambda_ws * min(1.0, max(0.0, (iteration - 5000) / 5000.0))
            curr_lambda_ov = min(args.lambda_ov, args.lambda_ov * (iteration / 5000.0))

            curr_lambda_face = args.lambda_face * min(1.0, max(0.0, (iteration - args.face_start) / max(1.0, float(args.face_warmup))))
            curr_lambda_cont = args.lambda_cont * min(1.0, max(0.0, (iteration - args.cont_start) / max(1.0, float(args.cont_warmup))))
            curr_lambda_anchor = args.lambda_anchor * min(1.0, max(0.0, (iteration - args.anchor_start) / max(1.0, float(args.anchor_warmup))))

            loss_G += (curr_lambda_ws * whitespace_style_loss(bbox_fake, padding_mask, wr_min=args.wr_min, style_mode=args.ws_style_mode, style_tags=style_tags)) + \
                      (curr_lambda_ov * pairwise_overlap_loss(bbox_fake, label, padding_mask)) + \
                      face_coverage_loss(bbox_fake, label, padding_mask, curr_lambda_face) + \
                      element_size_loss(bbox_fake, label, padding_mask, [IMG_ID], 0.35) + \
                      containment_loss(bbox_fake, label, padding_mask, lambda_cont=curr_lambda_cont) + \
                      face_anchor_loss(bbox_fake, label, padding_mask, face_pos, face_mask, lambda_anchor=curr_lambda_anchor) + \
                      (loss_aspect * 100.0) + (loss_min_size * 100.0)

            loss_G.backward()
            optimizerG.step()

            # Update D
            netD.zero_grad()
            loss_D = F.softplus(netD(bbox_fake.detach(), label, padding_mask)).mean() + \
                     F.softplus(-netD(bbox_real, label, padding_mask)).mean()
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

                with open(occ_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([iteration, float(occ_ratio_mean.item()), float(occ_rate.item()), n_face, n_occ])

                print(f'[{iteration}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} '
                      f'WS:{curr_lambda_ws:.3f} OV:{curr_lambda_ov:.3f} ALIGN:{curr_lambda_align:.3f} '
                      f'FACE:{curr_lambda_face:.3f} CONT:{curr_lambda_cont:.1f} ANCH:{curr_lambda_anchor:.1f} '
                      f'OCC:{occ_ratio.item():.3f} OCC_RATE:{occ_rate.item():.2f}')

            if iteration % args.vis_every == 0:
                netG.eval()
                with torch.no_grad():
                    # 生成時傳入完整的 fixed_mask，讓模型感知人臉位置
                    bbox_vis = torch.clamp(netG(fixed_z, fixed_label, ~fixed_mask), 0, 1)
                    
                    # 建立視覺化遮罩：保留 ID 0-2 與 ID 5 (Face)，排除 3 與 4
                    final_vis_mask = fixed_mask.clone()
                    for b in range(fixed_label.size(0)):
                        for n in range(fixed_label.size(1)):
                            fid = fixed_label[b, n].item()
                            final_vis_mask = fixed_mask & (fixed_label != BG_ID) & (fixed_label != MASK_ID) & (fixed_label != FACE_ID)
                    
                    vis_save_path = os.path.join(out_dir, f'fake_{iteration:05d}.png')
                    
                    colors_pil = [tuple(int(x) for x in c) for c in train_dataset.colors]
                    final_vis_mask = final_vis_mask & (fixed_label < len(colors_pil))
                    colors_pil = [tuple(int(x) for x in c) for c in train_dataset.colors]
                    save_image(bbox_vis, fixed_label, final_vis_mask, colors_pil, vis_save_path)

                    print(f"===> 已儲存預覽圖 (成功包含人臉框): {vis_save_path}")
                netG.train()
            iteration += 1

if __name__ == "__main__":
    main()