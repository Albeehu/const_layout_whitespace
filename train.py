import os
import argparse
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.data import DataLoader
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
    """
    boxes: (..., 4) in [cx, cy, w, h] (0~1)
    回傳:   (..., 4) in [x1, y1, x2, y2]
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


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

# ========= Face-aware helpers end =========1217 end

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
                        help='number of iterations to train for')
    parser.add_argument('--seed', type=int, help='manual seed')



    # General
    parser.add_argument('--latent_size', type=int, default=4,
                        help='latent size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--aug_flip', action='store_true',
                        help='use horizontal flip for data augmentation.')

    # Generator
    parser.add_argument('--G_d_model', type=int, default=256,
                        help='d_model for generator')
    parser.add_argument('--G_nhead', type=int, default=4,
                        help='nhead for generator')
    parser.add_argument('--G_num_layers', type=int, default=8,
                        help='num_layers for generator')

    # Discriminator
    parser.add_argument('--D_d_model', type=int, default=256,
                        help='d_model for discriminator')
    parser.add_argument('--D_nhead', type=int, default=4,
                        help='nhead for discriminator')
    parser.add_argument('--D_num_layers', type=int, default=8,
                        help='num_layers for discriminator')

    args = parser.parse_args()
    print(args)

    out_dir = init_experiment(args, "LayoutGAN++")
    writer = SummaryWriter(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    transforms = [LexicographicSort()]
    if args.aug_flip:
        transforms = [T.RandomApply([HorizontalFlip()], 0.5)] + transforms

    train_dataset = get_dataset(args.dataset, 'train',
                                transform=T.Compose(transforms))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)

    val_dataset = get_dataset(args.dataset, 'val')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False)

    num_label = train_dataset.num_classes

    # setup model
    netG = Generator(args.latent_size, num_label,
                     d_model=args.G_d_model,
                     nhead=args.G_nhead,
                     num_layers=args.G_num_layers,
                     ).to(device)

    netD = Discriminator(num_label,
                         d_model=args.D_d_model,
                         nhead=args.D_nhead,
                         num_layers=args.D_num_layers,
                         ).to(device)



    # prepare for evaluation
    # fid_train = LayoutFID(args.dataset, device)
    # fid_val = LayoutFID(args.dataset, device)

    # prepare for evaluation 12/8
    if args.dataset in ("crello_mainpart"):
        # v3 / mainpart / mainpart_face 版本：暫時不要用 FID，避免 layoutnet 類別數不匹配
        fid_train = None
        fid_val = None
    elif args.dataset == "crello_mainpart_face":
    # face 版本：用「crello」這個 layoutnet 權重來算 FID
        fid_train = LayoutFID("crello", device)
        fid_val   = LayoutFID("crello", device)
    else:
        fid_train = LayoutFID(args.dataset, device)
        fid_val = LayoutFID(args.dataset, device)


    fixed_label = None
    val_layouts = [(data.x.numpy(), data.y.numpy()) for data in val_dataset]

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)

    iteration = 0
    last_eval, best_iou = -1e+8, -1e+8
    max_epoch = args.iteration * args.batch_size / len(train_dataset)
    max_epoch = int(torch.ceil(torch.tensor(max_epoch)).item())
    for epoch in range(max_epoch):
        netG.train(), netD.train()
        for i, data in enumerate(train_dataloader):
            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox_real, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask
            z = torch.randn(label.size(0), label.size(1),
                            args.latent_size, device=device)

            # Update G network
            netG.zero_grad()
            #G 產生一堆 layout boxes（同一個元素 index 的 label 仍然是原本那個 class，包括 face=5）
            bbox_fake = netG(z, label, padding_mask)
            D_fake = netD(bbox_fake, label, padding_mask)
            #原本的 GAN 目標：讓 D 認為 fake 是 real
            loss_G = F.softplus(-D_fake).mean()
            # 1210 新增：避開人臉的 loss（非人臉元素壓到人臉就被懲罰）
            #找出label=5的所有ele. -> face boxes，label != 5的非face boxes，算兩者間的IoU，IoU大則loss大
            loss_face = face_coverage_loss(
                bbox_fake=bbox_fake,
                label=label,
                padding_mask=padding_mask,
            )
            # total loss = loss_G + face penalty
            #loss_G = loss_G + loss_face
            #反向傳遞 & 更新G
            loss_G.backward()
            optimizerG.step()

            # Update D network
            netD.zero_grad()
            D_fake = netD(bbox_fake.detach(), label, padding_mask)
            loss_D_fake = F.softplus(D_fake).mean()

            D_real, logit_cls, bbox_recon = \
                netD(bbox_real, label, padding_mask, reconst=True)
            loss_D_real = F.softplus(-D_real).mean()
            loss_D_recl = F.cross_entropy(logit_cls, data.y)
            loss_D_recb = F.mse_loss(bbox_recon, data.x)

            loss_D = loss_D_real + loss_D_fake
            loss_D += loss_D_recl + 10 * loss_D_recb
            loss_D.backward()
            optimizerD.step()

        #1216
        if fid_train is not None:
            # 先做一份「給 FID 用的 label 副本」
            label_for_fid = label
            padding_mask_for_fid = padding_mask

            if args.dataset == "crello_mainpart_face":
                # 專門給 FID 用的「忽略 face」版本
                # face = 5 的位置，對 FID 來說當成 padding，不進 layoutnet
                label_for_fid = label.clone()
                mask_face = (label_for_fid == FACE_ID)  # FACE_ID = 5

                padding_mask_for_fid = padding_mask | mask_face
                # 這些位置反正被當成 padding，不會用到，安全起見把值壓成 0 (合法範圍內)
                label_for_fid[mask_face] = 0

            fid_train.collect_features(bbox_fake, label_for_fid, padding_mask_for_fid)
            fid_train.collect_features(bbox_real, label_for_fid, padding_mask_for_fid, real=True)
        #1216 end

            if iteration % 50 == 0:
                D_real = torch.sigmoid(D_real).mean().item()
                D_fake = torch.sigmoid(D_fake).mean().item()
                loss_D, loss_G = loss_D.item(), loss_G.item()
                loss_D_fake, loss_D_real = loss_D_fake.item(), loss_D_real.item()
                loss_D_recl, loss_D_recb = loss_D_recl.item(), loss_D_recb.item()

                print('\t'.join([
                    f'[{epoch}/{max_epoch}][{i}/{len(train_dataloader)}]',
                    f'Loss_D: {loss_D:E}', f'Loss_G: {loss_G:E}',
                    f'Real: {D_real:.3f}', f'Fake: {D_fake:.3f}',
                ]))

                # add data to tensorboard
                tag_scalar_dict = {'real': D_real, 'fake': D_fake}
                writer.add_scalars('Train/D_value', tag_scalar_dict, iteration)
                writer.add_scalar('Train/Loss_D', loss_D, iteration)
                writer.add_scalar('Train/Loss_D_fake', loss_D_fake, iteration)
                writer.add_scalar('Train/Loss_D_real', loss_D_real, iteration)
                writer.add_scalar('Train/Loss_D_recl', loss_D_recl, iteration)
                writer.add_scalar('Train/Loss_D_recb', loss_D_recb, iteration)
                writer.add_scalar('Train/Loss_G', loss_G, iteration)

            if iteration % 5000 == 0:
                out_path = out_dir / f'real_samples.png'
                if not out_path.exists():
                    save_image(bbox_real, label, mask,
                               train_dataset.colors, out_path)

                if fixed_label is None:
                    fixed_label = label
                    fixed_z = z
                    fixed_mask = mask

                with torch.no_grad():
                    netG.eval()
                    out_path = out_dir / f'fake_samples_{iteration:07d}.png'
                    bbox_fake = netG(fixed_z, fixed_label, ~fixed_mask)
                    save_image(bbox_fake, fixed_label, fixed_mask,
                               train_dataset.colors, out_path)
                    netG.train()

            iteration += 1

        #12/1
        #fid_score_train = fid_train.compute_score()
        if fid_train is not None:
            fid_score_train = fid_train.compute_score()
        else:
            fid_score_train = 0.0  # 或者直接給 0，當作 placeholder


        if epoch != max_epoch - 1:
            if iteration - last_eval < 1e+4:
                continue

        # validation
        last_eval = iteration
        fake_layouts = []
        netG.eval(), netD.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                data = data.to(device)
                label, mask = to_dense_batch(data.y, data.batch)
                bbox_real, _ = to_dense_batch(data.x, data.batch)
                padding_mask = ~mask
                z = torch.randn(label.size(0), label.size(1),
                                args.latent_size, device=device)

                bbox_fake = netG(z, label, padding_mask)

            #1216
            if fid_val is not None:
                label_for_fid = label
                padding_mask_for_fid = padding_mask

                if args.dataset == "crello_mainpart_face":
                    label_for_fid = label.clone()
                    mask_face = (label_for_fid == FACE_ID)

                    padding_mask_for_fid = padding_mask | mask_face
                    label_for_fid[mask_face] = 0  # 壓到合法 index，反正被 mask 掉

                fid_val.collect_features(bbox_fake, label_for_fid, padding_mask_for_fid)
                fid_val.collect_features(bbox_real, label_for_fid, padding_mask_for_fid, real=True)
                #1216 end

                # collect generated layouts
                for j in range(label.size(0)):
                    _mask = mask[j]
                    b = bbox_fake[j][_mask].cpu().numpy()
                    l = label[j][_mask].cpu().numpy()
                    fake_layouts.append((b, l))

        #12/1
        if fid_val is not None:
            fid_score_val = fid_val.compute_score()
        else:
            fid_score_val = 0.0

        #fid_score_val = fid_val.compute_score()
        max_iou_val = compute_maximum_iou(val_layouts, fake_layouts)

        writer.add_scalar('Epoch', epoch, iteration)
        tag_scalar_dict = {'train': fid_score_train, 'val': fid_score_val}
        writer.add_scalars('Score/Layout FID', tag_scalar_dict, iteration)
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


if __name__ == "__main__":
    main()
