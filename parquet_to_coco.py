import os
import json
import glob
import io
from PIL import Image
import pandas as pd

# 輸入路徑
DATA_DIR = "/home/albee/const_layout/publaynet/data"
OUT_DIR = "data/dataset/publaynet/raw/publaynet"
IMG_OUT_DIR = os.path.join(OUT_DIR, "images")

# PubLayNet 5 類
CATEGORIES = [
    {"id": 1, "name": "text"},
    {"id": 2, "name": "title"},
    {"id": 3, "name": "list"},
    {"id": 4, "name": "table"},
    {"id": 5, "name": "figure"},
]

def convert_split(split_name, out_file):
    print(f"Converting {split_name}...")

    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, f"{split_name}-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split '{split_name}' in {DATA_DIR}")

    # 合併 parquet
    df = pd.concat([pd.read_parquet(pq) for pq in parquet_files], ignore_index=True)

    images = []
    annotations = []

    ann_id = 1
    seen_images = set()

    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    for _, row in df.iterrows():
        img_id = int(row["id"])   # parquet 裡的 "id" 就是 image_id

        if img_id not in seen_images:
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size

            # 存圖片
            img_path = os.path.join(IMG_OUT_DIR, f"{img_id}.jpg")
            img.save(img_path)

            images.append({
                "id": img_id,
                "file_name": f"images/{img_id}.jpg",
                "height": height,
                "width": width
            })
            seen_images.add(img_id)

        for ann in row["annotations"]:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(ann["category_id"]),
                "bbox": [float(x) for x in ann["bbox"]],
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(coco, f)

    print(f"✅ Saved COCO JSON to {out_file}, total images={len(images)}, annotations={len(annotations)}")

def main():
    convert_split("train", os.path.join(OUT_DIR, "train.json"))
    convert_split("validation", os.path.join(OUT_DIR, "val.json"))

if __name__ == "__main__":
    main()
