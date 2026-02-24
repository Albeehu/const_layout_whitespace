import json
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

import torch
import torchvision.utils as vutils
import torchvision.transforms as T


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed:", seed)


def init_experiment(args, prefix):
    if args.seed is None:
        args.seed = random.randint(0, 10000)

    set_seed(args.seed)

    if not args.name:
        args.name = datetime.now().strftime('%Y%m%d%H%M%S%f')

    out_dir = Path('output') / args.dataset / prefix / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'args.json'
    with json_path.open('w') as f:
        json.dump(vars(args), f, indent=2)

    return out_dir


def save_checkpoint(state, is_best, out_dir):
    out_path = Path(out_dir) / 'checkpoint.pth.tar'
    torch.save(state, out_path)

    if is_best:
        best_path = Path(out_dir) / 'model_best.pth.tar'
        shutil.copyfile(out_path, best_path)


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    # 這是讓圖片散開的關鍵轉換
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2

#2/10需還原整段
"""
def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new('RGB', (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2],
                       outline=color,
                       fill=c_fill)
    return img
"""

def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new('RGB', (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    # 計算面積並排序
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)), key=lambda i: area[i], reverse=True)

    # 重點：確保 for i in indices 這行與上方的 area 對齊（通常是 4 個空格）
    for i in indices:
        # 這裡必須定義為 bbox
        bbox = boxes[i] 
        raw_color = colors[labels[i]]
        color = tuple(raw_color)
        
        c_fill = color + (100,)
        
        # 這樣呼叫時才找得到 bbox
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox) 
        
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        
        if hasattr(color, "tolist"):  color = color.tolist()
        if color is not None and not isinstance(color, (int, str)):
            color = tuple(int(x) for x in color)

        if hasattr(c_fill, "tolist"): c_fill = c_fill.tolist()
        if c_fill is not None and not isinstance(c_fill, (int, str)):
            c_fill = tuple(int(x) for x in c_fill)

        draw.rectangle([x1, y1, x2, y2],
                    outline=color,
                    fill=None)
    return img




def save_image(batch_boxes, batch_labels, batch_mask,
               dataset_colors, out_path, canvas_size=(60, 40),
               nrow=None):
    # batch_boxes: [B, N, 4]
    # batch_labels: [B, N]
    # batch_mask: [B, N]

    imgs = []
    B = batch_boxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes = batch_boxes[i][mask_i]
        labels = batch_labels[i][mask_i]
        img = convert_layout_to_image(boxes, labels,
                                      dataset_colors,
                                      canvas_size)
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    vutils.save_image(image, out_path, normalize=False, nrow=nrow)
