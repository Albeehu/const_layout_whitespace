import os
import argparse
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from metric import LayoutFID, compute_maximum_iou
from model.layoutganpp import Generator, Discriminator
from data.util import LexicographicSort, HorizontalFlip
from util import init_experiment, save_image, save_checkpoint

# ========= Face-aware helpers (for face-avoidance loss) =========1217

FACE_ID = 5  # CrelloDataset 裡 "face" 的 label id


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2.0).clamp(0.0, 1.0)
    y1 = (cy - h / 2.0).clamp(0.0, 1.0)
    x2 = (cx + w / 2.0).clamp(0.0, 1.0)
    y2 = (cy + h / 2.0).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)

#算IoU
def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N, 4), boxes2: (M, 4)，座標都是 [x1, y1, x2, y2]
    回傳 IoU: (N, M)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)

#懲罰覆蓋face labelß
CONTAINER_IDS = {2, 3}  # ImageElement(2), ColoredBackground(3)

"""
a形狀是(K, 4), b is (M, 4) M=圖中face數量 K=圖中非face數量
每個box是xyxy = [x1, y1, x2, y2](左上角、右下角)
取 max x1 & y1 取重疊區域的左上角座標
取 min x2 & y2 取重疊區域的右下角座標
形狀都是(K, M)
inter_w inter_h是交集的寬高 如果兩個矩形沒重疊會發生負數 所以要clamp成0 -> 表無重疊
inter_w * inter_h -> 交集面積
"""
#算交集面積矩陣(K,M)
def box_intersection_area_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, :, 3], b[None, :, 3]) if False else torch.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    return inter_w * inter_h

"""
算每個box的w, h 不合理就clamp成0
eps是避免face算每個box的w, h 不合理就clamp成0, eps是避免face
回傳每個 box 的面積向量 (num_boxes,)
"""
def box_area_xyxy(xyxy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
    h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
    return (w * h).clamp(min=eps)

"""
bbox_fake: (B,N,4)，生成框 cxcywh (0~1)
label: (B,N) 類別 id
padding_mask: (B,N) True 表示這格是 padding(無效元素)
contain_thresh: 用來判斷「容器幾乎包含臉」的門檻(0.95 表示 95%)
lambda_face: 這個 loss 的權重
eps: 數值穩定
"""
def face_coverage_loss(
    bbox_fake: torch.Tensor,
    label: torch.Tensor,
    padding_mask: torch.Tensor,
    face_id: int = FACE_ID,
    lambda_face: float = 0.3,
    contain_thresh: float = 0.95,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    coverage(face) = sum_other inter_area(other, face) / area(face)

    但如果 other 是「容器類」(ImageElement=2, ColoredBackground=3)，且幾乎包含 face，
    則視為 face 在內容中，不算遮蔽 -> 該 pair 的 inter 設 0
    """
    #初始化累積器
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    total = torch.tensor(0.0, device=device) #累積每張圖的face loss
    count = 0 #有計算到loss的圖片數（避免 batch 裡全都沒 face 時除以 0）

    #逐張處理
    for b in range(B):
        valid = ~padding_mask[b] #valid 是 (N,)，表示這張圖哪些位置是真的元素（不是 padding）
        if valid.sum() == 0: #如果都是padding 就跳過
            continue

        boxes_b  = bbox_fake[b][valid]   # (Nv,4) cxcywh Nv 是這張圖有效元素數量 後面只對有效元素算遮蔽
        labels_b = label[b][valid]       # (Nv,)

        face_mask = (labels_b == face_id) #face_mask 是 (Nv,)
        if face_mask.sum() == 0: #沒face 跳過不扣分
            continue
        #如果只有 face 沒有其他元素，當然也沒人會遮 → 跳過
        other_mask = ~face_mask
        if other_mask.sum() == 0:
            continue
        
        #轉成xyxy算交集
        fb_xyxy = xywh_to_xyxy(boxes_b[face_mask])    # (M,4) face數量
        ob_xyxy = xywh_to_xyxy(boxes_b[other_mask])   # (K,4) 非face數量
        other_labels = labels_b[other_mask]           # (K,) 會用來找出哪些 other 是容器類

        #算所有other-face的交集面積矩陣
        inter = box_intersection_area_xyxy(ob_xyxy, fb_xyxy)  # (K,M) 第 k 個 other 和第 m 個 face 的交集面積
        face_area = box_area_xyxy(fb_xyxy, eps=eps)           # (M,)第 m 個 face 的面積

        # 容器免罰：ImageElement / ColoredBackground 幾乎包含 face 就不算遮蔽
        #建一個 (K,) 的布林 mask：other 裡哪些是容器（id 2 或 3）
        is_container = torch.zeros_like(other_labels, dtype=torch.bool)
        for cid in CONTAINER_IDS:
            is_container |= (other_labels == cid) #|= 是逐元素 OR，把多個類別併起來

        if is_container.any():
            inter_c = inter[is_container]                         # (Kc,M)只取「容器 vs face」的交集子矩陣
            contain_ratio = inter_c / face_area[None, :]          # (Kc,M)該face有多少%被這個容器覆蓋
            contain_mask = (contain_ratio > contain_thresh)       # (Kc,M) % > 0.95視為內容不算遮擋
            inter_c = inter_c * (~contain_mask).to(inter.dtype) #把那些 “被容器包含的 pair” 的交集設為 0 → 等於不扣分

            inter = inter.clone()
            inter[is_container] = inter_c #把處理後的交集矩陣放回去（保留非容器的交集不變）

        #coverage：把所有遮擋面積加總、除以 face 面積
        #inter.sum(dim=0)：對每個 face，把所有 other 的交集加起來 → 該 face 被遮的總面積
        covered = inter.sum(dim=0)                         # (M,)
        coverage = (covered / face_area).clamp(0.0, 1.0)   # (M,)

        #每張圖的 loss：平方後平均，再累積 ex.讓遮0.8比0.2懲罰大很多
        total += (coverage ** 2).mean()
        count += 1

    #batch 結尾：無 face 就回 0；否則平均再乘權重
    if count == 0:
        return total
    return lambda_face * (total / count)


# ele互相overlap的loss 1/17
def pairwise_overlap_loss(bbox_fake, label, padding_mask, eps=1e-5):
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    W = torch.ones((10, 10), device=device)
    W[3, :] = 0.0; W[:, 3] = 0.0
    W[5, 2] = 0.0; W[2, 5] = 0.0
    W[1, 5] = 5.0; W[5, 1] = 5.0
    W[1, 1] = 2.0
    W[1, 2] = 0.1; W[2, 1] = 0.1
    losses = []
    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() <= 1: continue
        xyxy = xywh_to_xyxy(bbox_fake[b][valid])
        labs = label[b][valid]
        inter = box_intersection_area_xyxy(xyxy, xyxy)
        inter.fill_diagonal_(0)
        area = box_area_xyxy(xyxy, eps=1e-4)
        denom = torch.minimum(area[:, None], area[None, :]).clamp(min=1e-4)
        overlap_ratio = (inter / denom).clamp(max=2.0)
        pair_weights = W[labs[:, None], labs[None, :]]
        loss_b = (overlap_ratio * pair_weights).triu(1).sum()
        num_pairs = (valid.sum() * (valid.sum() - 1)) / 2
        losses.append(loss_b / num_pairs.clamp(min=1))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
# 1/17 end

# ========= Face-aware helpers end =========1217 end

#1229 新增 whitespace loss 讓超過70%留白可生成出留白的圖 沒超過就生成原本的圖
def whitespace_style_loss(bbox_fake: torch.Tensor, padding_mask: torch.Tensor, wr_min: float = 0.7, w_style: float = 1.0) -> torch.Tensor:
    device = bbox_fake.device
    B, N, _ = bbox_fake.shape
    losses = []
    for b in range(B):
        valid = ~padding_mask[b]
        if valid.sum() == 0: continue
        boxes_xyxy = xywh_to_xyxy(bbox_fake[b][valid])
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        bw, bh = (x2 - x1).clamp(min=0.0), (y2 - y1).clamp(min=0.0)
        occ = (bw * bh).sum().clamp(0.0, 1.0)
        wr = 1.0 - occ
        if wr < wr_min:
            losses.append(w_style * (wr_min - wr)**2)
            continue
        L = x1.min().clamp(0.001, 0.999) 
        R = (1.0 - x2.max()).clamp(0.001, 0.999)
        T = y1.min().clamp(0.001, 0.999)
        B_m = (1.0 - y2.max()).clamp(0.001, 0.999)

        margins = torch.stack([L, R, T, B_m])
        mean_m = margins.mean()
        # 增加 eps 防止 std 為 0
        std_m = margins.std(unbiased=False) + 1e-6
        S_frame = (mean_m - 0.5 * std_m).clamp(0.0, 1.0)
        h_max, h_min = torch.stack([L, R]).max(), torch.stack([L, R]).min()
        v_max, v_min = torch.stack([T, B_m]).max(), torch.stack([T, B_m]).min()
        S_side = (h_max + 0.8 * (h_max - h_min) + 0.2 * torch.stack([T, B_m]).mean()).clamp(0,1)
        S_tb = (v_max + 0.8 * (v_max - v_min) + 0.2 * torch.stack([L, R]).mean()).clamp(0,1)
        S_corner = torch.sqrt(S_side * S_tb).clamp(0,1)
        S_style = torch.stack([S_frame, S_side, S_tb, S_corner]).max()
        losses.append(w_style * (1.0 - S_style))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

#1229 end


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    parser.add_argument('--dataset', type=str, default='rico',
                        choices=['rico', 'publaynet', 'magazine', 'crello', 'crello_mainpart', 'crello_mainpart_face'],
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--iteration', type=int, default=int(2e+5),
                        help='number of iterations (batches) to train for')
    parser.add_argument('--seed', type=int, help='manual seed')

    # General
    parser.add_argument('--latent_size', type=int, default=4,
                        help='latent size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--aug_flip', action='store_true',
                        help='use horizontal flip for data augmentation.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader workers')

    # Debug / logging
    parser.add_argument('--log_every', type=int, default=50,
                        help='print/tensorboard frequency (steps)')
    parser.add_argument('--vis_every', type=int, default=5000,
                        help='save fake_samples frequency (steps)')
    parser.add_argument('--eval_every', type=int, default=10000,
                        help='validation frequency in steps (checked at epoch boundary)')
    parser.add_argument('--fid_every', type=int, default=1,
                        help='collect FID features every N steps (1=every step)')
    parser.add_argument('--fixed_sample', type=str, default='fixed_sample.pt',
                        help='path to a shared fixed sample for visualization (same file => comparable across runs)')

    # Generator
    parser.add_argument('--G_d_model', type=int, default=256,
                        help='d_model for generator')
    parser.add_argument('--G_nhead', type=int, default=4,
                        help='nhead for generator')
    parser.add_argument('--G_num_layers', type=int, default=8,
                        help='num_layers for generator')

    # Whitespace / face / overlap
    parser.add_argument('--lambda_ws', type=float, default=0.05,
                        help='weight for whitespace style loss (0 = disable)')
    parser.add_argument('--wr_min', type=float, default=0.7,
                        help='only when whitespace ratio >= wr_min, apply preferred whitespace style')
    parser.add_argument('--lambda_ov', type=float, default=0.2,
                        help='weight for pairwise overlap loss')
    parser.add_argument('--lambda_face', type=float, default=0.05,
                        help='weight for face coverage loss')

    # Discriminator
    parser.add_argument('--D_d_model', type=int, default=256,
                        help='d_model for discriminator')
    parser.add_argument('--D_nhead', type=int, default=4,
                        help='nhead for discriminator')
    parser.add_argument('--D_num_layers', type=int, default=8,
                        help='num_layers for discriminator')

    args = parser.parse_args()
    print(args)

    # Repro
    if args.seed is not None:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    out_dir = init_experiment(args, "LayoutGAN++")
    writer = SummaryWriter(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    transforms = [LexicographicSort()]
    if args.aug_flip:
        transforms = [T.RandomApply([HorizontalFlip()], 0.5)] + transforms

    train_dataset = get_dataset(args.dataset, 'train', transform=T.Compose(transforms))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = get_dataset(args.dataset, 'val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    num_label = train_dataset.num_classes

    # setup model
    netG = Generator(
        args.latent_size,
        num_label,
        d_model=args.G_d_model,
        nhead=args.G_nhead,
        num_layers=args.G_num_layers,
    ).to(device)

    netD = Discriminator(
        num_label,
        d_model=args.D_d_model,
        nhead=args.D_nhead,
        num_layers=args.D_num_layers,
    ).to(device)

    # prepare for evaluation
    if args.dataset in ("crello_mainpart",):
        # v3 / mainpart：暫時不要用 FID，避免 layoutnet 類別數不匹配
        fid_train = None
        fid_val = None
        # face 版本：用「crello」這個 layoutnet 權重來算 FID
        fid_train = LayoutFID("crello", device)
        fid_val = LayoutFID("crello", device)
    else:
        fid_train = LayoutFID(args.dataset, device)
        fid_val = LayoutFID(args.dataset, device)

    val_layouts = [(data.x.numpy(), data.y.numpy()) for data in val_dataset]

    # Fixed sample (shared across runs) for comparable visualization
    fixed_path = Path(args.fixed_sample)
    if not fixed_path.is_absolute():
        fixed_path = Path.cwd() / fixed_path

    if fixed_path.exists():
        ck = torch.load(fixed_path, map_location='cpu')
        fixed_label = ck['label'].to(device)
        fixed_mask = ck['mask'].to(device)
        fixed_z = ck['z'].to(device)
        fixed_bbox_real = ck.get('bbox_real')
        if fixed_bbox_real is not None:
            fixed_bbox_real = fixed_bbox_real.to(device)
        print(f"[fixed_sample] loaded: {fixed_path}")
    else:
        # deterministic: take first val batch (shuffle=False) + seeded z
        data0 = next(iter(val_dataloader)).to(device)
        fixed_label, fixed_mask = to_dense_batch(data0.y, data0.batch)
        fixed_bbox_real, _ = to_dense_batch(data0.x, data0.batch)
        g = torch.Generator(device=device)
        g.manual_seed(args.seed if args.seed is not None else 0)
        fixed_z = torch.randn(
            fixed_label.size(0),
            fixed_label.size(1),
            args.latent_size,
            device=device,
            generator=g,
        )
        torch.save({
            'label': fixed_label.detach().cpu(),
            'mask': fixed_mask.detach().cpu(),
            'z': fixed_z.detach().cpu(),
            'bbox_real': fixed_bbox_real.detach().cpu(),
        }, fixed_path)
        print(f"[fixed_sample] created: {fixed_path}")

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)

    iteration = 0
    last_eval, best_iou = -1e+8, -1e+8

    steps_per_epoch = max(1, len(train_dataloader))
    import math
    max_epoch = int(math.ceil(args.iteration / steps_per_epoch))

    for epoch in range(max_epoch):
        netG.train(), netD.train()

        for i, data in enumerate(train_dataloader):
            if iteration >= args.iteration:
                break

            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox_real, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask

            z = torch.randn(label.size(0), label.size(1), args.latent_size, device=device)

            # =========================
            #   Update G network
            # =========================
            netG.zero_grad()

            bbox_fake = netG(z, label, padding_mask)  # (B, N, 4)
            D_fake_g = netD(bbox_fake, label, padding_mask)

            loss_G_adv = F.softplus(-D_fake_g).mean()

            loss_ws = whitespace_style_loss(
                bbox_fake=bbox_fake,
                padding_mask=padding_mask,
                wr_min=args.wr_min,
                w_style=1.0,
            )

            loss_face = face_coverage_loss(
                bbox_fake=bbox_fake,
                label=label,
                padding_mask=padding_mask,
                lambda_face=args.lambda_face,
            )

            loss_ov = pairwise_overlap_loss(bbox_fake, label, padding_mask)

            loss_G = loss_G_adv + args.lambda_ws * loss_ws + args.lambda_ov * loss_ov + loss_face

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=0.5)
            optimizerG.step()

            # =========================
            #   Update D network
            # =========================
            netD.zero_grad()

            D_fake = netD(bbox_fake.detach(), label, padding_mask)
            loss_D_fake = F.softplus(D_fake).mean()

            D_real, logit_cls, bbox_recon = netD(bbox_real, label, padding_mask, reconst=True)
            loss_D_real = F.softplus(-D_real).mean()
            loss_D_recl = F.cross_entropy(logit_cls, data.y)
            loss_D_recb = F.mse_loss(bbox_recon, data.x)

            loss_D = loss_D_real + loss_D_fake
            loss_D = loss_D + loss_D_recl + 10 * loss_D_recb
            loss_D.backward()
            optimizerD.step()

            # =========================
            #   FID collection (train)
            # =========================
            if fid_train is not None and (args.fid_every > 0) and (iteration % args.fid_every == 0):


                l_fid = label.clone(); p_fid = padding_mask.clone(); m_f = (l_fid == FACE_ID)
                p_fid = p_fid | m_f; max_i = fid_train.model.emb_label.num_embeddings - 1
                l_fid[m_f] = 0; l_fid = torch.clamp(l_fid, 0, max_i)
                fid_train.collect_features(bbox_fake.detach(), l_fid, p_fid)
                fid_train.collect_features(bbox_real, l_fid, p_fid, real=True)
                l_fid = label.clone(); p_fid = padding_mask.clone(); m_f = (l_fid == FACE_ID)
                p_fid = p_fid | m_f; max_i = fid_train.model.emb_label.num_embeddings - 1
                l_fid[m_f] = 0; l_fid = torch.clamp(l_fid, 0, max_i)
                fid_train.collect_features(bbox_fake.detach(), l_fid, p_fid)
                fid_train.collect_features(bbox_real, l_fid, p_fid, real=True)

            # =========================
            #   Logging
            # =========================
            if iteration % args.log_every == 0:
                D_real_p = torch.sigmoid(D_real.detach()).mean().item()
                D_fake_p = torch.sigmoid(D_fake.detach()).mean().item()

                print('	'.join([
                    f'[{epoch}/{max_epoch}][{i}/{len(train_dataloader)}]',
                    f'Loss_D: {loss_D.item():E}',
                    f'Loss_G: {loss_G.item():E}',
                    f'Real: {D_real_p:.3f}',
                    f'Fake: {D_fake_p:.3f}',
                ]))

                writer.add_scalars('Train/D_value', {'real': D_real_p, 'fake': D_fake_p}, iteration)
                writer.add_scalar('Train/Loss_D', loss_D.item(), iteration)
                writer.add_scalar('Train/Loss_D_fake', loss_D_fake.item(), iteration)
                writer.add_scalar('Train/Loss_D_real', loss_D_real.item(), iteration)
                writer.add_scalar('Train/Loss_D_recl', loss_D_recl.item(), iteration)
                writer.add_scalar('Train/Loss_D_recb', loss_D_recb.item(), iteration)

                writer.add_scalar('Train/Loss_G', loss_G.item(), iteration)
                writer.add_scalar('Train/Loss_G_adv', loss_G_adv.item(), iteration)
                writer.add_scalar('Train/Loss_ws', loss_ws.item(), iteration)
                writer.add_scalar('Train/Loss_face', loss_face.item(), iteration)
                writer.add_scalar('Train/Loss_ov', loss_ov.item(), iteration)

            # =========================
            #   Visualization
            # =========================
            # --- 修改後的視覺化邏輯 ---
            if iteration % args.vis_every == 0:
                hide_ids = [3, 4]

                if iteration == 0:
                    v = fixed_label[fixed_mask]
                    uniq, cnt = torch.unique(v, return_counts=True)
                    print("fixed labels:", list(zip(uniq.tolist(), cnt.tolist())))

                # hide_mask: True 表示該節點是 label 3/4
                hide_mask = torch.zeros_like(fixed_label, dtype=torch.bool)
                for hid in hide_ids:
                    hide_mask |= (fixed_label == hid)

                # 1) 給 Generator 的 padding_mask：只包含真正的 padding（不要 hide）
                gen_padding_mask = ~fixed_mask   # True=padding

                # 2) 給 visualization 的 padding_mask：把 3/4 也當成 padding（不畫）
                vis_padding_mask = gen_padding_mask | hide_mask

                if iteration == 0:
                    print(f"DEBUG: 只隱藏顯示 label {hide_ids}（不改生成）")

                # --- 1. Real Sample（只是不畫 3/4）---
                if fixed_bbox_real is not None:
                    out_path_real = out_dir / 'real_samples.png'
                    save_image(
                        fixed_bbox_real, fixed_label, vis_padding_mask,
                        train_dataset.colors, out_path_real
                    )

                # --- 2. Fake Sample（生成時仍包含 3/4，但存圖不畫 3/4）---
                with torch.no_grad():
                    netG.eval()
                    out_path_fake = out_dir / f'fake_samples_{iteration:07d}.png'

                    # 注意：這裡用 gen_padding_mask，不要用 vis_padding_mask
                    bbox_vis = netG(fixed_z, fixed_label, gen_padding_mask)

                    # 存圖才用 vis_padding_mask 隱藏 3/4
                    save_image(
                        bbox_vis, fixed_label, vis_padding_mask,
                        train_dataset.colors, out_path_fake
                    )
                    netG.train()

            iteration += 1

        # end epoch
        if fid_train is not None:
            fid_score_train = fid_train.compute_score()
        else:
            fid_score_train = 0.0

        # validation scheduling (checked at epoch boundary)
        if epoch != max_epoch - 1:
            if iteration - last_eval < args.eval_every and iteration < args.iteration:
                continue

        # validation
        last_eval = iteration
        fake_layouts = []
        netG.eval(), netD.eval()

        with torch.no_grad():
            for _, data in enumerate(val_dataloader):
                data = data.to(device)
                label, mask = to_dense_batch(data.y, data.batch)
                bbox_real, _ = to_dense_batch(data.x, data.batch)
                padding_mask = ~mask

                z = torch.randn(label.size(0), label.size(1), args.latent_size, device=device)
                bbox_fake = netG(z, label, padding_mask)

                if fid_val is not None:
                    # 1. 準備 FID 專用的標籤與 Mask
                    l_fv = label.clone()
                    p_fv = padding_mask.clone()
                    m_fv = (l_fv == FACE_ID)
                    
                    # 2. 屏蔽 Face 標籤防止越界
                    p_fv = p_fv | m_fv
                    max_iv = fid_val.model.emb_label.num_embeddings - 1
                    l_fv[m_fv] = 0
                    l_fv = torch.clamp(l_fv, 0, max_iv)

                    # 3. 傳入修正後的變數 (確保這裡的參數名稱與上方定義一致)
                    fid_val.collect_features(bbox_fake, l_fv, p_fv)
                    fid_val.collect_features(bbox_real, l_fv, p_fv, real=True)

                # collect generated layouts
                for j in range(label.size(0)):
                    _mask = mask[j]
                    b = bbox_fake[j][_mask].cpu().numpy()
                    l = label[j][_mask].cpu().numpy()
                    fake_layouts.append((b, l))

        if fid_val is not None:
            try:
                fid_score_val = fid_val.compute_score()
                print(f"[VAL] FID: {fid_score_val:.4f}")
            except ValueError as e:
                print(f"[WARN] FID 計算遇到數值問題（imaginary component），本輪先跳過：{e}")
                fid_score_val = float("nan")
        else:
            fid_score_val = 0.0

        max_iou_val = compute_maximum_iou(val_layouts, fake_layouts)

        writer.add_scalar('Epoch', epoch, iteration)
        writer.add_scalars('Score/Layout FID', {'train': fid_score_train, 'val': fid_score_val}, iteration)
        writer.add_scalar('Score/Maximum IoU', max_iou_val, iteration)

        # do checkpointing
        is_best = best_iou < max_iou_val
        best_iou = max(max_iou_val, best_iou)

        save_checkpoint({
            'args': vars(args),
            'epoch': epoch + 1,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'best_iou': best_iou,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
        }, is_best, out_dir)

        if iteration >= args.iteration:
            break

    writer.close()


if __name__ == "__main__":
    main()
