"""用 train_fixed_v6_0.6pkl_v17 改 v18
    1. 修改 RawLayoutDataset max_nodes = 15 -> max_nodes = 6
    2. RawLayoutDataset 的篩選條件改成若 ele. 元素不足時 ele選擇的的priority 
    3. face不列入max_nodes計數 如果有一樣會加進圖片中
    4. 改用'fixed_sample_v18.pt'

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
    1. Face (ID=5) 不計入 max_nodes=6 的配額。
    2. 其他元件依優先權（Image > Text > SVG > BG > Mask）保留最多 6 個。
    3. 移除 Face 面積排序，直接按原始順序取前 max_faces 個。
    4. 最後將 Face 合併回生成清單，確保生成器會輸出人臉框。
    """
    def __init__(self, data_list, num_classes, max_nodes=6, max_faces=4, colors=None):
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

        # ===== 1) 抽出 face 並準備作為固定約束 =====
        face_idx = (label == FACE_ID)
        f_label = label[face_idx]
        f_bbox = bbox[face_idx]
        
        # 移除面積排序，直接按原始順序截斷
        if len(f_label) > self.max_faces:
            f_label = f_label[:self.max_faces]
            f_bbox = f_bbox[:self.max_faces]

        face_pos = torch.zeros((self.max_faces, 4), dtype=torch.float)
        face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
        nf = len(f_label)
        if nf > 0:
            face_pos[:nf] = torch.FloatTensor(f_bbox)
            face_mask[:nf] = True

        # ===== 2) 處理其餘元件 (受 max_nodes=6 限制) =====
        o_label = label[~face_idx]
        o_bbox = bbox[~face_idx]

        # 大面積 SVG 映射為 Image (2)
        if len(o_label) > 0:
            area = o_bbox[:, 2] * o_bbox[:, 3]
            o_label[(o_label == 0) & (area > 0.15)] = IMG_ID

        # 優先權篩選 (排除 Face 後剩餘的元件選 6 個)
        n = len(o_label)
        if n > self.max_nodes:
            # 重新計算映射後的面積
            area = o_bbox[:, 2] * o_bbox[:, 3]
            idxs = np.arange(n)
            priority = np.ones(n, dtype=np.int64)
            priority[o_label == IMG_ID] = 0
            priority[o_label == TEXT_ID] = 1
            priority[o_label == SVG_ID] = 2
            priority[o_label == BG_ID] = 3
            priority[o_label == MASK_ID] = 4
            
            # 使用 lexsort，當 priority 相同時，保留面積大者優先
            order = np.lexsort((idxs, -area, priority))
            keep = order[:self.max_nodes]
            # 保持原始順序
            keep = np.sort(keep)
            
            o_label = o_label[keep]
            o_bbox = o_bbox[keep]

        # ===== 3) 合併：將最多 6 個元件與最多 4 個 Face 合併成最終生成清單 =====
        final_label = np.concatenate([o_label, f_label])
        final_bbox = np.concatenate([o_bbox, f_bbox])
        
        # 總長度上限為 6 + 4 = 10
        total_cap = self.max_nodes + self.max_faces
        pad_x = torch.full((total_cap,), self.num_classes - 1, dtype=torch.long)
        pad_pos = torch.zeros((total_cap, 4), dtype=torch.float)
        pad_mask = torch.zeros((total_cap,), dtype=torch.bool)

        curr_total = len(final_label)
        curr_total = len(final_label) # 合併後的總數量 (最多 6 + 4 = 10)
        pad_x[:curr_total] = torch.LongTensor(final_label)
        pad_pos[:curr_total] = torch.FloatTensor(final_bbox)
        pad_mask[:curr_total] = True  # 修正為總長度

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

def containment_loss(bbox_fake, label, padding_mask, face_pos, face_mask, outer_id=IMG_ID, lambda_cont=20.0):
    """
    讓 ground-truth face bbox 被某個 Image(outer_id) 容器包含（取最符合的容器做 min）。
    Face 不在 layout nodes 裡，因此從 face_pos/face_mask 讀取。
    """
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total_loss, count = torch.tensor(0.0, device=device), 0

    for b in range(B):
        valid_nodes = ~padding_mask[b]
        out_m = (label[b] == outer_id) & valid_nodes
        in_m = face_mask[b]

        if (not out_m.any()) or (not in_m.any()):
            continue

        out_xyxy = xywh_to_xyxy(bbox_fake[b][out_m])              # [M,4]
        in_xyxy  = xywh_to_xyxy(face_pos[b][in_m])               # [F,4]

        # 對每個 face，計算被每個容器「推出去」的距離懲罰，取 min 容器
        # out_xyxy: [M,4], in_xyxy: [F,4]
        # p*: [M,F]
        p1 = torch.relu(out_xyxy[:, 0:1] - in_xyxy[:, 0].T)
        p2 = torch.relu(in_xyxy[:, 2].T - out_xyxy[:, 2:3])
        p3 = torch.relu(out_xyxy[:, 1:2] - in_xyxy[:, 1].T)
        p4 = torch.relu(in_xyxy[:, 3].T - out_xyxy[:, 3:4])
        penalty = (p1 + p2 + p3 + p4)                            # [M,F]

        # 每個 face 找最好的容器
        total_loss += penalty.min(dim=0)[0].mean()
        count += 1

    return lambda_cont * (total_loss / count) if count > 0 else total_loss


def alignment_loss(bbox_fake, padding_mask, lambda_align=5.0):
    device = bbox_fake.device
    valid_boxes = bbox_fake[~padding_mask]
    if valid_boxes.size(0) == 0: return torch.tensor(0.0, device=device)
    c_pts = valid_boxes[:, :2]; dist = torch.abs(c_pts - 0.5)
    return lambda_align * dist.mean()

def element_size_loss(bbox_fake, label, padding_mask, target_ids, max_area, lambda_size=10.0):
    device = bbox_fake.device
    mask = torch.zeros_like(label, dtype=torch.bool)
    for tid in target_ids: mask |= (label == tid)
    mask &= (~padding_mask)
    if not mask.any(): return torch.tensor(0.0, device=device)
    areas = bbox_fake[:, :, 2] * bbox_fake[:, :, 3]
    return lambda_size * torch.relu(areas[mask] - max_area).mean()

def face_coverage_loss(bbox_fake, label, padding_mask, face_pos, face_mask, lambda_face=0.3):
    """
    懲罰「非容器元件」覆蓋到 ground-truth face bbox。
    - 排除 Image(2) / Background(3) / Mask(4) 覆蓋人臉（通常屬於容器或不重要元素）
    """
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total, count = torch.tensor(0.0, device=device), 0

    for b in range(B):
        in_m = face_mask[b]
        if not in_m.any():
            continue

        valid = ~padding_mask[b]
        # 只懲罰會「真的遮住臉」的元素
        other_valid = valid & ~( (label[b] == IMG_ID) | (label[b] == BG_ID) | (label[b] == MASK_ID) )
        if not other_valid.any():
            continue

        fb_xyxy = xywh_to_xyxy(face_pos[b][in_m])                 # [F,4]
        ob_xyxy = xywh_to_xyxy(bbox_fake[b][other_valid])         # [K,4]

        inter = box_intersection_area_xyxy(ob_xyxy, fb_xyxy)      # [K,F]
        face_area = box_area_xyxy(fb_xyxy)                        # [F]

        # 每個 face 的覆蓋比例（所有遮擋元素加總 / face_area）
        cover = (inter.sum(dim=0) / face_area).clamp(0, 1)
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
        (158, 218, 229),  # 5: face (淺紫色)
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
            loss_G_adv = F.softplus(-netD(bbox_fake, label, padding_mask)).mean()
            
            # 方案 1：延遲啟動
            curr_lambda_ws = args.lambda_ws * min(1.0, max(0.0, (iteration - 5000) / 5000.0))
            loss_ws = whitespace_style_loss(bbox_fake, padding_mask, args.wr_min)
            
            loss_face = face_coverage_loss(bbox_fake, label, padding_mask, face_pos, face_mask, lambda_face=args.lambda_face)
            loss_contain = containment_loss(bbox_fake, label, padding_mask, face_pos, face_mask, outer_id=IMG_ID)
            loss_align = alignment_loss(bbox_fake, padding_mask)
            
            # 尺寸限制：針對 ID 2(圖片容器) 與其他裝飾
            loss_img_size = element_size_loss(bbox_fake, label, padding_mask, [IMG_ID], 0.35)
            loss_svg_size = element_size_loss(bbox_fake, label, padding_mask, [0, BG_ID], 0.15)
            
            curr_lambda_ov = min(args.lambda_ov, args.lambda_ov * (iteration / 5000.0))
            loss_ov = pairwise_overlap_loss(bbox_fake, label, padding_mask)

            # --- [新增] 文字 (ID 1) 避開人臉 (GT face_pos) 的強效 Loss ---
            loss_face_text_avoid = 0.0
            fake_xyxy = xywh_to_xyxy(bbox_fake)  # [B, N, 4]

            for b in range(label.size(0)):
                v = ~padding_mask[b]
                t_idx = (label[b] == 1) & v  # 文字 ID = 1
                f_idx = face_mask[b]         # GT face slots

                if t_idx.any() and f_idx.any():
                    face_xyxy = xywh_to_xyxy(face_pos[b][f_idx])  # [F,4]
                    inter = box_intersection_area_xyxy(fake_xyxy[b][t_idx], face_xyxy)  # [T,F]
                    loss_face_text_avoid += inter.sum() * 15.0

            loss_face_text_avoid = loss_face_text_avoid / label.size(0)

            # 更新總 Loss
            loss_G = loss_G_adv + (curr_lambda_ws * loss_ws) + (curr_lambda_ov * loss_ov) + \
                     loss_face + loss_img_size + loss_svg_size + loss_contain + loss_align + \
                     loss_face_text_avoid  # <--- 加入這一項

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