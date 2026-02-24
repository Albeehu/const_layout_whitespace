"""用 train_fixed_v6_0.6pkl_v34 改 v35
    1. 修改 RawLayoutDataset、containment_loss、loss_G 計算 
"""
import os
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
        bbox, label = np.array(bbox), np.array(label)

        # 1. 分離 Face 與其餘元件
        face_mask_idx = (label == FACE_ID)
        temp_f_label, temp_f_bbox = label[face_mask_idx], bbox[face_mask_idx]
        o_label, o_bbox = label[~face_mask_idx], bbox[~face_mask_idx]

        # 2. 映射大面積 SVG 為 Image (ID=2)
        if len(o_label) > 0:
            area = o_bbox[:, 2] * o_bbox[:, 3]
            o_label[(o_label == 0) & (area > 0.15)] = IMG_ID

        # 3. 優先權過濾 (最多 4 個節點)
        n = len(o_label)
        if n > self.max_nodes:
            area = o_bbox[:, 2] * o_bbox[:, 3]
            priority = np.ones(n, dtype=np.int64)
            priority[o_label == IMG_ID] = 0
            priority[o_label == TEXT_ID] = 1
            order = np.lexsort((np.arange(n), -area, priority))
            keep = np.sort(order[:self.max_nodes])
            o_label, o_bbox = o_label[keep], o_bbox[keep]

        # 4. [優化] 直接保留所有人臉標註，不進行 is_contained 過濾，為 Loss 提供連續信號
        f_label, f_bbox = temp_f_label, temp_f_bbox
        face_pos = torch.zeros((self.max_faces, 4)); face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
        nf = min(len(f_label), self.max_faces)
        if nf > 0:
            face_pos[:nf] = torch.FloatTensor(f_bbox[:nf])
            face_mask[:nf] = True

        # 5. 合併與 Padding
        final_label = np.concatenate([o_label, f_label[:nf]])
        final_bbox = np.concatenate([o_bbox, f_bbox[:nf] if nf > 0 else np.empty((0,4))])
        
        cap = self.max_nodes + self.max_faces
        pad_x = torch.full((cap,), self.num_classes - 1, dtype=torch.long)
        pad_pos = torch.zeros((cap, 4)); pad_mask = torch.zeros((cap,), dtype=torch.bool)
        
        pad_x[:len(final_label)] = torch.LongTensor(final_label)
        pad_pos[:len(final_label)] = torch.FloatTensor(final_bbox)
        pad_mask[:len(final_label)] = True 

        return {'x': pad_x, 'pos': pad_pos, 'mask': pad_mask, 'face_pos': face_pos, 'face_mask': face_mask}



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

def whitespace_style_loss(bbox_fake, padding_mask, wr_min=0.6, w_style=1.0):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    losses = []
    margin = 0.05  # 緩衝區方案
    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() == 0: continue
        x1, y1, x2, y2 = xywh_to_xyxy(bbox_fake[b][valid]).unbind(-1)
        bw, bh = (x2 - x1).clamp(min=0.0), (y2 - y1).clamp(min=0.0)
        wr = 1.0 - (bw * bh).sum().clamp(0.0, 1.0)
        if wr < (wr_min - margin):
            losses.append(w_style * (wr_min - margin - wr)**2); continue
        L, R, T, B_m = x1.min().clamp(0.001, 0.999), (1.0 - x2.max()).clamp(0.001, 0.999), y1.min().clamp(0.001, 0.999), (1.0 - y2.max()).clamp(0.001, 0.999)
        margins = torch.stack([L, R, T, B_m])
        S_frame = (margins.mean() - 0.5 * (margins.std(unbiased=False) + 1e-6)).clamp(0.0, 1.0)
        h_max, v_max = torch.stack([L, R]).max(), torch.stack([T, B_m]).max()
        S_side = (h_max + 0.8 * (h_max - torch.stack([L, R]).min()) + 0.2 * torch.stack([T, B_m]).mean()).clamp(0,1)
        S_tb = (v_max + 0.8 * (v_max - torch.stack([T, B_m]).min()) + 0.2 * torch.stack([L, R]).mean()).clamp(0,1)
        S_style = torch.stack([S_frame, S_side, S_tb, torch.sqrt(S_side * S_tb).clamp(0,1)]).max()
        losses.append(w_style * (1.0 - S_style))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

def containment_loss(bbox_fake, label, padding_mask, lambda_cont=150.0):
    device = bbox_fake.device
    B, total_loss, count = bbox_fake.shape[0], torch.tensor(0.0, device=device), 0
    for b in range(B):
        valid = ~padding_mask[b]
        out_m, in_m = (label[b] == IMG_ID) & valid, (label[b] == FACE_ID) & valid
        if not (out_m.any() and in_m.any()): continue
        out_xyxy, in_xyxy = xywh_to_xyxy(bbox_fake[b][out_m]), xywh_to_xyxy(bbox_fake[b][in_m])
        # 1. 面積包含比例懲罰
        inter = box_intersection_area_xyxy(out_xyxy, in_xyxy)
        ratio = (inter / box_area_xyxy(in_xyxy)).max(dim=0)[0]
        # 2. [新增] 物理中心吸引力：強迫人臉中心靠近圖片中心
        img_center = bbox_fake[b][out_m][:, :2].mean(dim=0)
        face_center = bbox_fake[b][in_m][:, :2]
        dist_loss = F.mse_loss(face_center, img_center.expand_as(face_center))
        total_loss += torch.relu(0.98 - ratio).mean() + dist_loss * 2.0
        count += 1
    return lambda_cont * (total_loss / count) if count > 0 else total_loss

def alignment_loss(bbox_fake, padding_mask, lambda_align=5.0):
    device = bbox_fake.device
    valid_boxes = bbox_fake[~padding_mask]
    if valid_boxes.size(0) == 0: return torch.tensor(0.0, device=device)
    c_pts = valid_boxes[:, :2]; dist = torch.abs(c_pts - 0.5)
    return lambda_align * dist.mean()

#強化文字對齊損失
def grid_alignment_loss(bbox_fake, padding_mask, lambda_grid=10.0):
    # 強迫元件的左邊界或中心點靠近 0.25, 0.5, 0.75 等線條
    device = bbox_fake.device
    valid = bbox_fake[~padding_mask]
    if valid.size(0) == 0: return torch.tensor(0.0, device=device)
    
    x1, _, x2, _ = xywh_to_xyxy(valid).unbind(-1)
    centers = (x1 + x2) / 2
    
    # 計算與最近 0.1 步長網格的距離
    grid = torch.tensor([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], device=device)
    dist_l = torch.abs(x1[:, None] - grid[None, :]).min(dim=1)[0].mean()
    dist_c = torch.abs(centers[:, None] - grid[None, :]).min(dim=1)[0].mean()
    
    return lambda_grid * (dist_l + dist_c)

def element_size_loss(bbox_fake, label, padding_mask, target_ids, max_area, lambda_size=10.0):
    device = bbox_fake.device
    mask = torch.zeros_like(label, dtype=torch.bool)
    for tid in target_ids: mask |= (label == tid)
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


def pairwise_overlap_loss(bbox_fake, label, padding_mask):
    device = bbox_fake.device
    W = torch.ones((10, 10), device=device)
    W[BG_ID, :], W[:, BG_ID], W[MASK_ID, :], W[:, MASK_ID] = 0.0, 0.0, 0.0, 0.0
    W[IMG_ID, FACE_ID], W[FACE_ID, IMG_ID] = 0.0, 0.0 
    W[1, 1], W[1, FACE_ID], W[FACE_ID, 1] = 25.0, 100.0, 100.0 # 文字互壓與文字壓臉重罰
    
    losses = []
    for b in range(bbox_fake.size(0)):
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
    parser.add_argument('--dataset', type=str, default='crello')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iteration', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--lambda_ws', type=float, default=0.08)
    parser.add_argument('--wr_min', type=float, default=0.6)
    parser.add_argument('--lambda_ov', type=float, default=1.5)
    parser.add_argument('--lambda_face', type=float, default=1.0)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--aug_flip', action='store_true')
    parser.add_argument('--vis_every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fixed_sample', type=str, default='fixed_sample_v18.pt') # 確保檔名正確
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 資料載入
    # 1. 獲取原始 dataset
    train_dataset = get_dataset(args.dataset, 'train')
    
    # 2. 如果有生成的 pkl 資料，重新封裝
    pkl_path = "output/crello/LayoutGAN++/crello_ws_gt0_6/crello_ws_gt0_6_generated.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f: 
            high_quality_data = pickle.load(f)
        train_dataset = RawLayoutDataset(high_quality_data, num_classes=train_dataset.num_classes, colors=train_dataset.colors)
    
    # --- [關鍵修正：將顏色轉換為 PIL 可接受的 0-255 整數格式] ---
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
    fixed_path = 'fixed_sample_v18.pt'
    if not os.path.exists(fixed_path):
        print(f"致命錯誤：找不到 {fixed_path}。請先執行產生腳本產出二進位檔案。")
        exit()

    # 確保使用 map_location 以相容不同設備
    ck = torch.load(fixed_path, weights_only=False, map_location=device)
    fixed_label = ck['label'].to(device)
    fixed_mask = ck['mask'].to(device)
    fixed_z = ck['z'].to(device)
    
    # Face 不作為 layout node：在視覺化時過濾掉 ID 5
    fixed_mask_noface = fixed_mask & (fixed_label != FACE_ID)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    


    iteration = 0
    for epoch in range(1000):
        for data in train_dataloader:
            if iteration >= args.iteration: break
            label, pos, mask = data['x'].to(device), data['pos'].to(device), data['mask'].to(device)
            face_pos, face_mask = data['face_pos'].to(device), data['face_mask'].to(device)
            bbox_real = pos
            padding_mask = ~mask
            curr_lambda_ws = args.lambda_ws * min(1.0, max(0.0, (iteration - 5000) / 5000.0))

            netG.train(); netG.zero_grad()
            bbox_fake = torch.clamp(netG(torch.randn(label.size(0), label.size(1), 4, device=device), label, padding_mask), 0, 1)
            
            # --- [正確的 Loss 加總邏輯] ---
            loss_G = F.softplus(-netD(bbox_fake, label, padding_mask)).mean()
            valid = ~padding_mask

            # 1. [核心：元件高度與面積保護] 防止縮成一條線
            fake_wh = bbox_fake[valid][:, 2:]
            loss_shape_protect = torch.relu(0.18 - fake_wh[:, 1]).mean() + torch.relu(0.04 - (fake_wh[:, 0]*fake_wh[:, 1])).mean()
            loss_G += (loss_shape_protect * 180.0)

            # 2. [多樣風格引導：邊距門檻化] 
            style_idx = (iteration // 1500) % 4
            x1_f, x2_f = (bbox_fake[:,:,0]-bbox_fake[:,:,2]/2), (bbox_fake[:,:,0]+bbox_fake[:,:,2]/2)
            y2_f = (bbox_fake[:,:,1]+bbox_fake[:,:,3]/2)
            
            loss_style = torch.tensor(0.0, device=device)
            if style_idx == 0: # 左右排版：獎勵左邊距 L > 0.3
                loss_style = torch.relu(0.3 - x1_f[mask].min())
            elif style_idx == 3: # 角落排版：獎勵元件底邊壓低
                loss_style = torch.abs(y2_f[mask] - 0.92).mean()
            loss_G += (loss_style * 120.0)

            # 3. 基礎美學損失
            curr_ws = args.lambda_ws * min(1.0, max(0.0, (iteration - 5000) / 5000.0))
            loss_G += whitespace_style_loss(bbox_fake, padding_mask, args.wr_min) * curr_ws
            loss_G += containment_loss(bbox_fake, label, padding_mask, lambda_cont=250.0)
            
            loss_G.backward(); optimizerG.step()

            # Update D
            netD.zero_grad()
            loss_D = F.softplus(netD(bbox_fake.detach(), label, padding_mask)).mean() + \
                     F.softplus(-netD(bbox_real, label, padding_mask)).mean()
            loss_D.backward(); optimizerD.step()

            if iteration % 100 == 0:
                print(f'[{iteration}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} WS_Weight: {curr_lambda_ws:.3f}')

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
                            if fid == BG_ID or fid == MASK_ID:
                                final_vis_mask[b, n] = False
                    
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