"""用 train_fixed_v6_0.6pkl_v26 改 v27
    1. 強迫生成的圖片面積至少要達到畫布的 10%

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

            # 3. 清洗人臉數據
            final_f_label = []
            final_f_bbox = []
            img_boxes = o_bbox[o_label == IMG_ID] # 這是過濾後的 4 個節點中的圖片

            for i in range(len(temp_f_label)):
                fb = temp_f_bbox[i]
                is_contained = False
                for ib in img_boxes:
                    # 判斷臉部中心點 (fb[0], fb[1]) 是否在圖片框 ib 內
                    if (fb[0] >= ib[0] - ib[2]/2 and fb[0] <= ib[0] + ib[2]/2 and
                        fb[1] >= ib[1] - ib[3]/2 and fb[1] <= ib[1] + ib[3]/2):
                        is_contained = True
                        break
                if is_contained:
                    final_f_label.append(FACE_ID)
                    final_f_bbox.append(fb)

            # 確保 final_f_bbox 轉換為 numpy 時維度正確
            if len(final_f_label) > 0:
                f_label = np.array(final_f_label)
                f_bbox = np.array(final_f_bbox)
            else:
                f_label = np.array([], dtype=np.int64)
                f_bbox = np.empty((0, 4)) 

            # 4. 初始化回傳給 Face Anchor Loss 的數據
            # 關鍵：現在 face_mask 只會對「真正有效的臉」標記為 True
            face_pos = torch.zeros((self.max_faces, 4), dtype=torch.float)
            face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
            nf = len(f_label)
            if nf > 0:
                face_pos[:nf] = torch.FloatTensor(f_bbox)
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
            netG.zero_grad()
            bbox_fake = torch.clamp(netG(z, label, padding_mask), 0, 1)
            loss_G = F.softplus(-netD(bbox_fake, label, padding_mask)).mean()
            loss_G_adv = F.softplus(-netD(bbox_fake, label, padding_mask)).mean()

            valid_mask = ~padding_mask
            img_mask = (label == IMG_ID) & valid_mask

            if img_mask.any():
                # 強迫生成的圖片面積至少要達到畫布的 10% (0.1)
                img_areas = bbox_fake[img_mask][:, 2] * bbox_fake[img_mask][:, 3]
                # 權重必須設得比空白損失高很多
                loss_img_min_area = torch.relu(0.10 - img_areas).mean()
                loss_G += (loss_img_min_area * 150.0) # 強制圖片長大，讓人臉有家可回
            
            # --- [關鍵修正：定義延遲啟動邏輯與空白損失] ---
            # 方案 1：空白損失延遲啟動 (5000 iter 後開始增加權重)
            curr_lambda_ws = args.lambda_ws * min(1.0, max(0.0, (iteration - 5000) / 5000.0))
            loss_ws = whitespace_style_loss(bbox_fake, padding_mask, args.wr_min)
            
            # 方案 2：重疊損失延遲啟動
            curr_lambda_ov = min(args.lambda_ov, args.lambda_ov * (iteration / 5000.0))
            loss_ov = pairwise_overlap_loss(bbox_fake, label, padding_mask)
            
            # --- [人臉相關約束] ---
            # 1. 人臉避讓
            loss_face = face_coverage_loss(bbox_fake, label, padding_mask, lambda_face=args.lambda_face)
            
            # 2. 包含約束 (提升權重至 50.0)
            loss_contain = containment_loss(bbox_fake, label, padding_mask, outer_id=IMG_ID, inner_id=FACE_ID, lambda_cont=50.0)
            
            # 3. 錨定約束 (對齊真實座標)
            loss_anchor = face_anchor_loss(bbox_fake, label, padding_mask, face_pos, face_mask, lambda_anchor=50.0)

            # --- [新增：長寬比約束] ---
            # 懲罰過於細長的元件 (例如長寬比超過 8 倍或小於 1/8)
            # 使用 bbox_fake 的 w (index 2) 和 h (index 3)
            fake_wh = bbox_fake[~padding_mask][:, 2:]
            aspect_ratio = fake_wh[:, 0] / (fake_wh[:, 1] + 1e-6)
            # 這裡設定當寬度是高度的 8 倍以上，或高度是寬度的 8 倍以上時產生懲罰
            loss_aspect = (torch.relu(aspect_ratio - 8.0) + torch.relu(0.125 - aspect_ratio)).mean()
            
            # --- [關鍵新增：最小面積約束] ---
            # 取得所有有效元件的面積 (w * h)
            areas = bbox_fake[~padding_mask][:, 2] * bbox_fake[~padding_mask][:, 3]
            # 如果面積小於 0.02 (畫布的 5%)，則產生懲罰
            loss_min_size = torch.relu(0.05 - areas).mean()

            # --- [其他美學約束] ---
            loss_align = alignment_loss(bbox_fake, padding_mask)
            loss_img_size = element_size_loss(bbox_fake, label, padding_mask, [IMG_ID], 0.35)
            loss_svg_size = element_size_loss(bbox_fake, label, padding_mask, [0, BG_ID], 0.15)
            
            # --- [最終總 Loss 加總] ---
            loss_G = loss_G_adv + (curr_lambda_ws * loss_ws) + (curr_lambda_ov * loss_ov) + \
                     loss_face + loss_img_size + loss_svg_size + loss_contain + loss_align + \
                     loss_anchor+ (loss_aspect * 50.0)+ (loss_min_size * 100.0)

            loss_G.backward()
            optimizerG.step()

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