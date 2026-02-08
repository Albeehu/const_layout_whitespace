#刪除train_fixed_v6_0.6pkl_v3_face 中的bbox_vis[face] = fixed_bbox_real[face]
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

# ========= Face-aware helpers =========
FACE_ID = 5  

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

def face_coverage_loss(bbox_fake, label, padding_mask, face_id=FACE_ID, lambda_face=0.3, contain_thresh=0.95, eps=1e-8):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total = torch.tensor(0.0, device=device)
    count = 0
    CONTAINER_IDS = {2, 3}

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

def pairwise_overlap_loss(bbox_fake, label, padding_mask, eps=1e-5):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    W = torch.ones((10, 10), device=device)
    W[3, :], W[:, 3], W[5, 2], W[2, 5] = 0.0, 0.0, 0.0, 0.0
    W[1, 5], W[5, 1], W[1, 1], W[1, 2], W[2, 1] = 5.0, 5.0, 2.0, 0.1, 0.1
    losses = []
    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() <= 1: continue
        xyxy = xywh_to_xyxy(bbox_fake[b][valid]); labs = label[b][valid]
        inter = box_intersection_area_xyxy(xyxy, xyxy); inter.fill_diagonal_(0)
        area = box_area_xyxy(xyxy, eps=1e-4)
        denom = torch.minimum(area[:, None], area[None, :]).clamp(min=1e-4)
        overlap_ratio = (inter / denom).clamp(max=2.0)
        loss_b = (overlap_ratio * W[labs[:, None], labs[None, :]]).triu(1).sum()
        losses.append(loss_b / ((valid.sum() * (valid.sum() - 1)) / 2).clamp(min=1))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

def whitespace_style_loss(bbox_fake, padding_mask, wr_min=0.7, w_style=1.0):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    losses = []
    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() == 0: continue
        x1, y1, x2, y2 = xywh_to_xyxy(bbox_fake[b][valid]).unbind(-1)
        bw, bh = (x2 - x1).clamp(min=0.0), (y2 - y1).clamp(min=0.0)
        wr = 1.0 - (bw * bh).sum().clamp(0.0, 1.0)
        if wr < wr_min:
            losses.append(w_style * (wr_min - wr)**2); continue
        L, R, T, B_m = x1.min().clamp(0.001, 0.999), (1.0 - x2.max()).clamp(0.001, 0.999), y1.min().clamp(0.001, 0.999), (1.0 - y2.max()).clamp(0.001, 0.999)
        margins = torch.stack([L, R, T, B_m])
        S_frame = (margins.mean() - 0.5 * (margins.std(unbiased=False) + 1e-6)).clamp(0.0, 1.0)
        h_max = torch.stack([L, R]).max(); v_max = torch.stack([T, B_m]).max()
        S_side = (h_max + 0.8 * (h_max - torch.stack([L, R]).min()) + 0.2 * torch.stack([T, B_m]).mean()).clamp(0,1)
        S_tb = (v_max + 0.8 * (v_max - torch.stack([T, B_m]).min()) + 0.2 * torch.stack([L, R]).mean()).clamp(0,1)
        S_style = torch.stack([S_frame, S_side, S_tb, torch.sqrt(S_side * S_tb).clamp(0,1)]).max()
        losses.append(w_style * (1.0 - S_style))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='crello')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iteration', type=int, default=int(2e+5))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--aug_flip', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--vis_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--fid_every', type=int, default=1)
    parser.add_argument('--fixed_sample', type=str, default='fixed_sample.pt')
    parser.add_argument('--G_d_model', type=int, default=256)
    parser.add_argument('--G_nhead', type=int, default=4)
    parser.add_argument('--G_num_layers', type=int, default=8)
    parser.add_argument('--lambda_ws', type=float, default=0.05)
    parser.add_argument('--wr_min', type=float, default=0.7)
    parser.add_argument('--lambda_ov', type=float, default=0.2)
    parser.add_argument('--lambda_face', type=float, default=0.05)
    parser.add_argument('--D_d_model', type=int, default=256)
    parser.add_argument('--D_nhead', type=int, default=4)
    parser.add_argument('--D_num_layers', type=int, default=8)
    args = parser.parse_args()

    out_dir = init_experiment(args, "LayoutGAN++")
    writer = SummaryWriter(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transforms = [LexicographicSort()]
    if args.aug_flip: transforms = [T.RandomApply([HorizontalFlip()], 0.5)] + transforms

    train_dataset = get_dataset(args.dataset, 'train', transform=T.Compose(transforms))
    pkl_path = "output/crello/LayoutGAN++/crello_ws_gt0_6/crello_ws_gt0_6_generated.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f: high_quality_data = pickle.load(f)
        orig_dataset = get_dataset(args.dataset, 'train')
        train_dataset = RawLayoutDataset(high_quality_data, num_classes=orig_dataset.num_classes, colors=getattr(orig_dataset, 'colors', None))
        print(f"成功加載高品質數據，樣本數：{len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_dataset = get_dataset(args.dataset, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    num_label = train_dataset.num_classes if hasattr(train_dataset, 'num_classes') else train_dataset.dataset.num_classes
    netG = Generator(args.latent_size, num_label, d_model=args.G_d_model, nhead=args.G_nhead, num_layers=args.G_num_layers).to(device)
    netD = Discriminator(num_label, d_model=args.D_d_model, nhead=args.D_nhead, num_layers=args.D_num_layers).to(device)
    fid_train = LayoutFID(args.dataset, device); fid_val = LayoutFID(args.dataset, device)

    fixed_path = Path(args.fixed_sample)
    if fixed_path.exists():
        ck = torch.load(fixed_path, map_location='cpu')
        fixed_label, fixed_mask, fixed_z = ck['label'].to(device), ck['mask'].to(device), ck['z'].to(device)
        fixed_bbox_real = ck.get('bbox_real').to(device) if ck.get('bbox_real') is not None else None
    else:
        from torch_geometric.utils import to_dense_batch
        data0 = next(iter(val_dataloader)).to(device)
        fixed_label, fixed_mask = to_dense_batch(data0.y, data0.batch)
        fixed_bbox_real, _ = to_dense_batch(data0.x, data0.batch)
        fixed_z = torch.randn(fixed_label.size(0), fixed_label.size(1), args.latent_size, device=device)
        torch.save({'label': fixed_label.cpu(), 'mask': fixed_mask.cpu(), 'z': fixed_z.cpu(), 'bbox_real': fixed_bbox_real.cpu()}, fixed_path)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr); optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
    iteration, last_eval, best_iou = 0, -1e+8, -1e+8

    FACE_ID = 5
    HIDE_IDS = [3, 4]
    
    for epoch in range(int(1e+5)):
        netG.train(); netD.train()
        for i, data in enumerate(train_dataloader):
            if iteration >= args.iteration: break
            
            # --- 資料分流處理 ---
            if isinstance(data, dict):
                data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
                label, pos, mask = data['x'], data['pos'], data['mask']
            else:
                from torch_geometric.utils import to_dense_batch
                data = data.to(device)
                label, mask = to_dense_batch(data.y, data.batch)
                pos, _ = to_dense_batch(data.x, data.batch)
            
            bbox_real, padding_mask = pos, ~mask
            z = torch.randn(label.size(0), label.size(1), args.latent_size, device=device)

            netG.zero_grad()
            bbox_fake = netG(z, label, padding_mask)
            loss_G_adv = F.softplus(-netD(bbox_fake, label, padding_mask)).mean()
            loss_ws = whitespace_style_loss(bbox_fake, padding_mask, args.wr_min)
            loss_face = face_coverage_loss(bbox_fake, label, padding_mask, lambda_face=args.lambda_face)
            loss_ov = pairwise_overlap_loss(bbox_fake, label, padding_mask)
            loss_G = loss_G_adv + args.lambda_ws * loss_ws + args.lambda_ov * loss_ov + loss_face
            loss_G.backward(); torch.nn.utils.clip_grad_norm_(netG.parameters(), 0.5); optimizerG.step()

            # =========================
            #   Update D network
            # =========================
            netD.zero_grad()

            D_fake = netD(bbox_fake.detach(), label, padding_mask)
            D_real, logit_cls, bbox_recon = netD(bbox_real, label, padding_mask, reconst=True)
            
            loss_D = F.softplus(D_fake).mean() + F.softplus(-D_real).mean()
            
            # --- 修正 Reconstruction Loss ---
            target_y = data.y if hasattr(data, 'y') else data['x']
            
            if isinstance(data, dict):
                # 如果是高品質子集，我們必須根據 mask 過濾標籤，使維度與 logit_cls 一致
                # mask 是 [B, 50]，取出 True 的部分
                valid_target_y = target_y[mask] 
                loss_D += F.cross_entropy(logit_cls, valid_target_y, ignore_index=-1)
                
                # Bbox 重建損失也需要過濾
                valid_bbox_real = bbox_real[mask]
                loss_D += 10 * F.mse_loss(bbox_recon, valid_bbox_real)
            else:
                # 原始 Crello 格式
                loss_D += F.cross_entropy(logit_cls.view(-1, num_label), target_y.view(-1), ignore_index=-1)
                loss_D += 10 * F.mse_loss(bbox_recon, data.x)

            loss_D.backward()
            optimizerD.step()

            if iteration % args.fid_every == 0:
                l_f, p_f = label.clone(), padding_mask.clone(); m = (l_f == FACE_ID); p_f |= m
                l_f[m] = 0; l_f = torch.clamp(l_f, 0, fid_train.model.emb_label.num_embeddings - 1)
                fid_train.collect_features(bbox_fake.detach(), l_f, p_f); fid_train.collect_features(bbox_real, l_f, p_f, real=True)

            if iteration % args.log_every == 0:
                print(f'[{epoch}][{i}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}')
                writer.add_scalar('Train/Loss_G', loss_G.item(), iteration)
                writer.add_scalar('Train/Loss_ws', loss_ws.item(), iteration)

            #0205 生成face label
            if iteration % args.vis_every == 0:
                netG.eval()
                with torch.no_grad():
                    bbox_vis = netG(fixed_z, fixed_label, ~fixed_mask)

                    # 用 GT bbox 固定 face（只當參考區域）
                    if fixed_bbox_real is not None:
                        face = (fixed_label == FACE_ID) & fixed_mask
                        bbox_vis = bbox_vis.clone()
                        #bbox_vis[face] = fixed_bbox_real[face]

                    hide_mask = torch.zeros_like(fixed_label, dtype=torch.bool)
                    for hid in HIDE_IDS:
                        hide_mask |= (fixed_label == hid)

                    vis_mask = fixed_mask & (~hide_mask)   # 只影響「畫不畫」
                    save_image(bbox_vis, fixed_label, vis_mask, train_dataset.colors,
                            out_dir / f'fake_{iteration:07d}.png')
                netG.train()
                #0205 end

            iteration += 1

        # Validation
        if iteration - last_eval >= args.eval_every:
            last_eval, fake_layouts = iteration, []
            netG.eval(); netD.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    from torch_geometric.utils import to_dense_batch
                    data = data.to(device); label, mask = to_dense_batch(data.y, data.batch)
                    pos, _ = to_dense_batch(data.x, data.batch); p_mask = ~mask
                    bbox_fake = netG(torch.randn(label.size(0), label.size(1), args.latent_size, device=device), label, p_mask)
                    for j in range(label.size(0)):
                        fake_layouts.append((bbox_fake[j][mask[j]].cpu().numpy(), label[j][mask[j]].cpu().numpy()))
            max_iou_val = compute_maximum_iou([(data.x.numpy(), data.y.numpy()) for data in val_dataset], fake_layouts)
            writer.add_scalar('Score/Maximum IoU', max_iou_val, iteration)
            save_checkpoint({'netG': netG.state_dict(), 'netD': netD.state_dict(), 'best_iou': max_iou_val}, best_iou < max_iou_val, out_dir)
            best_iou = max(max_iou_val, best_iou)

    writer.close()

if __name__ == "__main__":
    main()