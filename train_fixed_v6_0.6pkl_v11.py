"""用 train_fixed_v6_0.6pkl_v10 改 v11
    1. 實作留白指標延遲啟動 (Solution 1)
    2. 實作留白緩衝區 (Solution 4)
    3. 保留 v10 的原始標籤命名，避免 NameError
"""
# train_fixed_v6_0.6pkl_v11.py
import os
import argparse
import pickle
from pathlib import Path
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

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

# ========= Constants (保留你原本 v10 的命名) =========
FACE_ID = 5  
SVG_ID = 2   # 這是你 v10 原本定義的
BG_ID = 3

class RawLayoutDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, num_classes, max_nodes=50, colors=None):
        self.data_list = data_list
        self.num_classes = num_classes
        self.max_nodes = max_nodes  
        self.colors = colors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        bbox, label = self.data_list[idx]
        n = len(label)
        curr_n = min(n, self.max_nodes)
        
        pad_x = torch.full((self.max_nodes,), self.num_classes - 1, dtype=torch.long)
        pad_pos = torch.zeros((self.max_nodes, 4), dtype=torch.float)
        pad_mask = torch.zeros((self.max_nodes,), dtype=torch.bool)

        pad_x[:curr_n] = torch.LongTensor(label[:curr_n])
        pad_pos[:curr_n] = torch.FloatTensor(bbox[:curr_n])
        pad_mask[:curr_n] = True

        return {
            'x': pad_x,
            'pos': pad_pos,
            'mask': pad_mask
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
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    return inter_w * inter_h

def box_area_xyxy(xyxy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
    h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
    return (w * h).clamp(min=eps)

def face_coverage_loss(bbox_fake, label, padding_mask, face_id=5, lambda_face=0.3, contain_thresh=0.95, eps=1e-8):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total = torch.tensor(0.0, device=device)
    count = 0
    CONTAINER_IDS = {2, 3} # 依照你的 label 順序

    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() == 0: continue
        boxes_b = bbox_fake[b][valid]
        labels_b = label[b][valid]
        face_mask = (labels_b == face_id)
        if face_mask.sum() == 0: continue
        other_mask = ~face_mask
        if other_mask.sum() == 0: continue
        fb_xyxy = xywh_to_xyxy(boxes_b[face_mask]); ob_xyxy = xywh_to_xyxy(boxes_b[other_mask])
        other_labels = labels_b[other_mask]
        inter = box_intersection_area_xyxy(ob_xyxy, fb_xyxy); face_area = box_area_xyxy(fb_xyxy, eps=eps)
        is_container = torch.zeros_like(other_labels, dtype=torch.bool)
        for cid in CONTAINER_IDS: is_container |= (other_labels == cid)
        if is_container.any():
            inter_c = inter[is_container]; contain_mask = (inter_c / face_area[None, :] > contain_thresh)
            inter[is_container] = inter_c * (~contain_mask).to(inter.dtype)
        coverage = (inter.sum(dim=0) / face_area).clamp(0.0, 1.0)
        total += (coverage ** 2).mean(); count += 1
    return lambda_face * (total / count) if count > 0 else total

def element_size_loss(bbox_fake, label, padding_mask, target_ids, max_area, lambda_size=5.0):
    device = bbox_fake.device
    mask = torch.zeros_like(label, dtype=torch.bool)
    for tid in target_ids: mask |= (label == tid)
    mask &= (~padding_mask)
    if not mask.any(): return torch.tensor(0.0, device=device)
    areas = bbox_fake[:, :, 2] * bbox_fake[:, :, 3]
    size_penalty = torch.relu(areas[mask] - max_area).mean()
    return lambda_size * size_penalty

def containment_loss(bbox_fake, label, padding_mask, inner_id=5, outer_id=2, lambda_cont=20.0):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total_loss, count = torch.tensor(0.0, device=device), 0
    for b in range(B):
        valid = ~padding_mask[b]
        in_m, out_m = (label[b] == inner_id) & valid, (label[b] == outer_id) & valid
        if not in_m.any() or not out_m.any(): continue
        in_xyxy, out_xyxy = xywh_to_xyxy(bbox_fake[b][in_m]), xywh_to_xyxy(bbox_fake[b][out_m])
        p1 = torch.relu(out_xyxy[:, 0:1] - in_xyxy[:, 0].T) 
        p2 = torch.relu(in_xyxy[:, 2].T - out_xyxy[:, 2:3]) 
        p3 = torch.relu(out_xyxy[:, 1:2] - in_xyxy[:, 1].T) 
        p4 = torch.relu(in_xyxy[:, 3].T - out_xyxy[:, 3:4])
        total_loss += (p1 + p2 + p3 + p4).min(dim=0)[0].mean(); count += 1
    return lambda_cont * (total_loss / count) if count > 0 else total_loss

def alignment_loss(bbox_fake, padding_mask, lambda_align=5.0):
    device = bbox_fake.device
    valid_boxes = bbox_fake[~padding_mask]
    if valid_boxes.size(0) == 0: return torch.tensor(0.0, device=device)
    c_pts = valid_boxes[:, :2]; dist = torch.abs(c_pts - 0.5)
    return lambda_align * dist.mean()

def pairwise_overlap_loss(bbox_fake, label, padding_mask):
    device = bbox_fake.device
    W = torch.ones((10, 10), device=device)
    # 背景(3)不懲罰
    W[3, :], W[:, 3] = 0.0, 0.0
    # 臉(5)與圖片(2)不懲罰
    W[2, 5], W[5, 2] = 0.0, 0.0
    
    losses = []
    for b in range(bbox_fake.size(0)):
        valid = ~padding_mask[b]
        if valid.sum() <= 1: continue
        xyxy = xywh_to_xyxy(bbox_fake[b][valid]); labs = label[b][valid]
        inter = box_intersection_area_xyxy(xyxy, xyxy); inter.fill_diagonal_(0)
        area = box_area_xyxy(xyxy, eps=1e-4)
        overlap_ratio = (inter / torch.minimum(area[:, None], area[None, :]).clamp(min=1e-4)).clamp(max=2.0)
        loss_b = (overlap_ratio * W[labs[:, None], labs[None, :]]).triu(1).sum()
        losses.append(loss_b / ((valid.sum() * (valid.sum() - 1)) / 2).clamp(min=1))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

# 方案 4：留白緩衝區
def whitespace_style_loss(bbox_fake, padding_mask, wr_min=0.6, w_style=1.0):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    losses = []
    margin = 0.05 
    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() == 0: continue
        x1, y1, x2, y2 = xywh_to_xyxy(bbox_fake[b][valid]).unbind(-1)
        bw, bh = (x2 - x1).clamp(min=0.0), (y2 - y1).clamp(min=0.0)
        wr = 1.0 - (bw * bh).sum().clamp(0.0, 1.0)
        
        if wr < (wr_min - margin):
            losses.append(w_style * (wr_min - margin - wr)**2)
            continue
            
        L, R, T, B_m = x1.min().clamp(0.001, 0.999), (1.0 - x2.max()).clamp(0.001, 0.999), y1.min().clamp(0.001, 0.999), (1.0 - y2.max()).clamp(0.001, 0.999)
        margins = torch.stack([L, R, T, B_m])
        S_frame = (margins.mean() - 0.5 * (margins.std(unbiased=False) + 1e-6)).clamp(0.0, 1.0)
        h_max, v_max = torch.stack([L, R]).max(), torch.stack([T, B_m]).max()
        S_side = (h_max + 0.8 * (h_max - torch.stack([L, R]).min()) + 0.2 * torch.stack([T, B_m]).mean()).clamp(0,1)
        S_tb = (v_max + 0.8 * (v_max - torch.stack([T, B_m]).min()) + 0.2 * torch.stack([L, R]).mean()).clamp(0,1)
        S_style = torch.stack([S_frame, S_side, S_tb, torch.sqrt(S_side * S_tb).clamp(0,1)]).max()
        losses.append(w_style * (1.0 - S_style))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='v11_fix')
    parser.add_argument('--dataset', type=str, default='crello')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iteration', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--lambda_ws', type=float, default=0.08)
    parser.add_argument('--wr_min', type=float, default=0.6)
    parser.add_argument('--lambda_ov', type=float, default=1.5)
    parser.add_argument('--lambda_face', type=float, default=1.0)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--aug_flip', action='store_true') # 注意指令要用 --aug_flip
    parser.add_argument('--vis_every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fixed_sample', type=str, default='fixed_sample.pt')
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--G_d_model', type=int, default=256)
    parser.add_argument('--G_nhead', type=int, default=4)
    parser.add_argument('--G_num_layers', type=int, default=8)
    parser.add_argument('--D_d_model', type=int, default=256)
    parser.add_argument('--D_nhead', type=int, default=4)
    parser.add_argument('--D_num_layers', type=int, default=8)
    args = parser.parse_args()

    out_dir = init_experiment(args, "LayoutGAN++")
    writer = SummaryWriter(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 資料讀取邏輯 (保留你的 v10)
    train_dataset = get_dataset(args.dataset, 'train')
    pkl_path = "output/crello/LayoutGAN++/crello_ws_gt0_6/crello_ws_gt0_6_generated.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f: high_quality_data = pickle.load(f)
        train_dataset = RawLayoutDataset(high_quality_data, num_classes=train_dataset.num_classes, colors=train_dataset.colors)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    netG = Generator(args.latent_size, train_dataset.num_classes, d_model=args.G_d_model, nhead=args.G_nhead, num_layers=args.G_num_layers).to(device)
    netD = Discriminator(train_dataset.num_classes, d_model=args.D_d_model, nhead=args.D_nhead, num_layers=args.D_num_layers).to(device)

    fixed_path = Path(args.fixed_sample)
    ck = torch.load(fixed_path); fixed_label, fixed_mask, fixed_z = ck['label'].to(device), ck['mask'].to(device), ck['z'].to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr); optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    iteration = 0
    for epoch in range(10000):
        for data in train_dataloader:
            if iteration >= args.iteration: break
            label, pos, mask = data['x'].to(device), data['pos'].to(device), data['mask'].to(device)
            bbox_real, padding_mask = pos, ~mask
            z = torch.randn(label.size(0), label.size(1), args.latent_size, device=device)

            # Update G
            netG.zero_grad()
            bbox_fake = torch.clamp(netG(z, label, padding_mask), 0, 1)
            loss_G_adv = F.softplus(-netD(bbox_fake, label, padding_mask)).mean()
            
            # 方案 1：延遲啟動
            curr_lambda_ws = args.lambda_ws * min(1.0, max(0.0, (iteration - 5000) / 5000.0))
            loss_ws = whitespace_style_loss(bbox_fake, padding_mask, args.wr_min)
            
            loss_face = face_coverage_loss(bbox_fake, label, padding_mask, lambda_face=args.lambda_face)
            loss_contain = containment_loss(bbox_fake, label, padding_mask, FACE_ID, 2)
            loss_align = alignment_loss(bbox_fake, padding_mask)
            
            loss_img_size = element_size_loss(bbox_fake, label, padding_mask, [2], 0.35)
            loss_svg_size = element_size_loss(bbox_fake, label, padding_mask, [0, 3], 0.15)
            
            curr_lambda_ov = min(args.lambda_ov, args.lambda_ov * (iteration / 5000.0))
            loss_ov = pairwise_overlap_loss(bbox_fake, label, padding_mask)

            loss_G = loss_G_adv + (curr_lambda_ws * loss_ws) + (curr_lambda_ov * loss_ov) + \
                     loss_face + loss_img_size + loss_svg_size + loss_contain + loss_align
            
            face_mask_g = (label == FACE_ID) & (~padding_mask)
            if face_mask_g.any(): loss_G += 5.0 * F.mse_loss(bbox_fake[face_mask_g], bbox_real[face_mask_g])

            loss_G.backward(); optimizerG.step()

            # Update D
            netD.zero_grad()
            loss_D = F.softplus(netD(bbox_fake.detach(), label, padding_mask)).mean() + \
                     F.softplus(-netD(bbox_real, label, padding_mask)).mean()
            loss_D.backward(); optimizerD.step()

            if iteration % 100 == 0:
                print(f'[{iteration}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} WS: {curr_lambda_ws:.3f}')

            if iteration % args.vis_every == 0:
                netG.eval()
                with torch.no_grad():
                    bbox_vis = netG(fixed_z, fixed_label, ~fixed_mask)
                    bbox_vis = torch.clamp(bbox_vis, 0, 1)

                    # 同時排除背景 (3) 和 遮罩 (4)
                    final_vis_mask = fixed_mask.clone()
                    # 建議直接使用你定義的變數名稱，或直接填入數字
                    for hid in [3, 4]: 
                        final_vis_mask &= (fixed_label != hid)
                    
                    save_image(bbox_vis, fixed_label, final_vis_mask, train_dataset.colors,
                            out_dir / f'fake_{iteration:07d}.png')
                netG.train()
            iteration += 1

if __name__ == "__main__":
    main()