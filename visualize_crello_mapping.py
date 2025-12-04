#確認新的pkl出來的layout是否可以對上png
from pathlib import Path
import pickle
from PIL import Image, ImageDraw

PKL_ROOT = Path("data/dataset/crello")
IMAGE_DIRS = {
    "train": Path("crello_v4_train_images"),
    "validation": Path("crello_v4_validation_images"),
    "test": Path("crello_v4_test_images"),
}

def visualize_one(split: str, layout_idx: int, out_path: str):
    # 讀 layout pkl & indices
    with (PKL_ROOT / f"crello_{split}_imgmap.pkl").open("rb") as f:
        layouts = pickle.load(f)
    with (PKL_ROOT / f"crello_{split}_imgmap_indices.pkl").open("rb") as f:
        hf_indices = pickle.load(f)

    boxes, labels = layouts[layout_idx]
    hf_idx = hf_indices[layout_idx]

    # 你的 png 命名是 000000.png 這種
    img_dir = IMAGE_DIRS[split]
    img_path = img_dir / f"{hf_idx:06d}.png"
    print("使用圖片:", img_path)

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    for (cx, cy, w, h), cls in zip(boxes, labels):
        # boxes 現在是 [cx, cy, w, h]，全部是「相對畫布比例」
        x1 = (cx - w / 2.0) * W
        y1 = (cy - h / 2.0) * H
        x2 = (cx + w / 2.0) * W
        y2 = (cy + h / 2.0) * H
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    img.save(out_path)
    print("已輸出疊框結果到:", out_path)

if __name__ == "__main__":
    # 先試幾張 validation 看看
    for i in range(0, 5):
        visualize_one("validation", layout_idx=i, out_path=f"val_{i:04d}_overlay.png")
