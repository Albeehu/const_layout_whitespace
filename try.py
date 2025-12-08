#!/usr/bin/env python
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

# 👇 跟 train.py 一樣的 import 方式
from data.crello import CrelloDataset


def draw_layout(sample, label_names, colors, size=512, face_id=5):
    """
    sample: 一個 PyG Data（有 x, y）
      - x: (N, 4)  [cx, cy, w, h] in 0~1
      - y: (N,)
    label_names: dataset.labels
    colors: dataset.colors (list of (r,g,b))
    """
    boxes = sample.x
    labels = sample.y

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # 可選：字型，如果沒有就讓它 fallback
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = None

    for box, c in zip(boxes, labels):
        cx, cy, w, h = box.tolist()

        # 假設是 [cx, cy, w, h]（YOLO-style）
        x1 = (cx - w / 2.0) * size
        y1 = (cy - h / 2.0) * size
        x2 = (cx + w / 2.0) * size
        y2 = (cy + h / 2.0) * size

        cls_id = int(c.item())
        color = colors[cls_id]

        # face 可以畫粗一點
        width = 3 if cls_id == face_id else 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # 左上角寫一點 label 文字
        name = label_names[cls_id] if cls_id < len(label_names) else str(cls_id)
        text = f"{cls_id}:{name}"
        if font is not None:
            draw.text((x1 + 2, y1 + 2), text, fill=color, font=font)
        else:
            draw.text((x1 + 2, y1 + 2), text, fill=color)

    return img


def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "debug_face_layouts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 用有 face 的 crello mainpart 版本
    ds = CrelloDataset(
        split="validation",   # 或 "val" / "test"，看你要看哪個 split
        transform=None,
        variant="default",
        use_face=True,
    )

    print("Dataset size:", len(ds))
    print("Labels:", ds.labels)

    # 隨機挑幾張（例如 5 張）
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:5]

    for idx in indices:
        sample = ds[idx]
        img = draw_layout(sample, ds.labels, ds.colors, size=512, face_id=5)
        out_path = out_dir / f"val_{idx:04d}.png"
        img.save(out_path)
        print("saved:", out_path)

    print("Done. 看看 debug_face_layouts/ 底下的 png 吧！")


if __name__ == "__main__":
    main()