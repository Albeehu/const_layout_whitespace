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

#1205
FACE_ID = 5  # CrelloDataset 裡 face 的 label id

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (..., 4) in cx, cy, w, h (0~1)
    回傳:   (..., 4) in x1, y1, x2, y2
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N, 4), boxes2: (M, 4)，x1,y1,x2,y2
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


def face_overlap_loss(pred_boxes_list, face_boxes_list, lambda_face: float = 0.3) -> torch.Tensor:
    """
    pred_boxes_list: list[Tensor(N_i, 4)] 或 Tensor(B, N, 4)，cx,cy,w,h 0~1
    face_boxes_list: list[Tensor(M_i, 4)]，每張圖對應的 GT 人臉 box

    回傳 scalar loss：生成的 box 跟人臉 IoU 越大，loss 越大。
    """
    if isinstance(pred_boxes_list, torch.Tensor):
        # 假設 pred_boxes_list: (B, N, 4)
        pred_boxes_list = [pb for pb in pred_boxes_list]

    device = pred_boxes_list[0].device
    total = torch.tensor(0.0, device=device)
    count = 0

    for pred_boxes, face_boxes in zip(pred_boxes_list, face_boxes_list):
        if pred_boxes.numel() == 0 or face_boxes.numel() == 0:
            continue

        pb_xyxy = xywh_to_xyxy(pred_boxes)
        fb_xyxy = xywh_to_xyxy(face_boxes.to(device))

        ious = box_iou_xyxy(pb_xyxy, fb_xyxy)      # (N, M)
        max_iou_per_pred, _ = ious.max(dim=1)      # (N,)

        # IoU^2 當懲罰，偏重高 IoU 的情況
        loss_img = (max_iou_per_pred ** 2).mean()

        total += loss_img
        count += 1

    if count == 0:
        return total  # 這個 batch 都沒有臉，loss=0

    return lambda_face * (total / count)
#1205 end

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
    if args.dataset in ("crello_mainpart", "crello_mainpart_face"):
        # v3 / mainpart / mainpart_face 版本：暫時不要用 FID，避免 layoutnet 類別數不匹配
        fid_train = None
        fid_val = None
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
            bbox_fake = netG(z, label, padding_mask)
            D_fake = netD(bbox_fake, label, padding_mask)
            loss_G = F.softplus(-D_fake).mean()
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

        if fid_train is not None:
            fid_train.collect_features(bbox_fake, label, padding_mask)
            fid_train.collect_features(bbox_real, label, padding_mask,
                                       real=True)

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

            if fid_val is not None:
                fid_val.collect_features(bbox_fake, label, padding_mask)
                fid_val.collect_features(bbox_real, label, padding_mask,
                                         real=True)

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
