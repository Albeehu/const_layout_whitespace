#生成新的pkl，用來與png對應，才能多＋一個label

# 讀 HuggingFace cyberagent/crello v4.0.0
# 根據 crello_v4_*_images/ 底下的 000000.png, 000001.png ...
# 生成新的 pkl：list[(boxes, labels)]
#   boxes: (N, 4)  -> [cx, cy, w, h] (全部是 normalized 到 canvas)
#   labels: (N,)   -> [0..4] 對應 Svg/Text/Image/ColoredBackground/SvgMask
#
# 檔名 000000.png -> ds[0], 000001.png -> ds[1], ...

from pathlib import Path
import pickle

import numpy as np
from datasets import load_dataset  # pip install datasets

# pkl 要存的地方
PKL_ROOT = Path("data/dataset/crello")

# 每個 split 對應到實際圖片資料夾
IMAGE_DIRS = {
    "train": Path("crello_v4_train_images"),
    "validation": Path("crello_v4_validation_images"),
    "test": Path("crello_v4_test_images"),
}

# 用 Crello v4，因為你的圖片是 v4 split render 出來的
CRELLO_REV = "4.0.0"


def build_split_pkl(split: str):
    print(f"\n=== 處理 split = {split} ===")

    # 1. 讀 HuggingFace crello v4 split
    ds = load_dataset("cyberagent/crello", revision=CRELLO_REV, split=split)
    type_names = ds.features["type"].feature.names
    print("HF type names:", type_names)
    print("HF", split, "size =", len(ds))

    # v4: canvas_width / canvas_height 是 class_label，不過我們這版不再用來做 normalization，
    # 只在需要時可以查真正 pixel size。
    cw_feat = ds.features["canvas_width"]
    ch_feat = ds.features["canvas_height"]

    # 2. 讀本地圖片
    img_dir = IMAGE_DIRS[split]
    png_files = sorted(img_dir.glob("*.png"))
    if not png_files:
        print(f"[警告] {img_dir} 底下沒有 png，略過 {split}")
        return

    print(f"本地 {split} png 檔數量 = {len(png_files)}")

    layouts = []      # list of (boxes, labels)
    img_indices = []  # 對應 HF dataset 的 index

    for png_path in png_files:
        stem = png_path.stem  # '000000', '000001', ...
        try:
            idx = int(stem)   # 000000 -> 0, 000001 -> 1, ...
        except ValueError:
            print(f"[警告] 檔名不是純數字，跳過: {png_path}")
            continue

        if idx < 0 or idx >= len(ds):
            print(f"[警告] idx={idx} 超出 HF {split} 範圍，跳過: {png_path}")
            continue

        ex = ds[idx]

        # 如果你之後要畫在 pixel 上可以用這兩個：
        W = float(cw_feat.int2str(ex["canvas_width"]))
        H = float(ch_feat.int2str(ex["canvas_height"]))

        lefts = ex["left"]     # 已經是 normalized 比例 (0~1)
        tops = ex["top"]
        widths = ex["width"]
        heights = ex["height"]
        types = ex["type"]     # list[int]，0~4 = Svg/Text/Image/ColoredBackground/SvgMask

        boxes = []
        labels = []

        for left, top, w, h, t_id in zip(lefts, tops, widths, heights, types):
            # Crello 給的是 [left, top, width, height] (normalized)
            # 轉成你原本 pkl 的格式 [cx, cy, w, h]
            cx = left + w / 2.0
            cy = top + h / 2.0
            boxes.append([cx, cy, w, h])
            labels.append(int(t_id))

        if not boxes:
            print(f"[警告] idx={idx} ({png_path.name}) 沒有任何 element，被略過")
            continue

        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        layouts.append((boxes, labels))
        img_indices.append(idx)

    print(f"{split} 實際收集到的 layout 筆數 = {len(layouts)}")

    PKL_ROOT.mkdir(parents=True, exist_ok=True)

    # 3. 存成新的 pkl，不跟原本撞名：*_imgmap.pkl
    out_pkl = PKL_ROOT / f"crello_{split}_imgmap.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(layouts, f)
    print(f"已寫出 {out_pkl}")

    # 4. 再存 HF index list：*_imgmap_indices.pkl
    out_idx = PKL_ROOT / f"crello_{split}_imgmap_indices.pkl"
    with out_idx.open("wb") as f:
        pickle.dump(img_indices, f)
    print(f"已寫出 {out_idx}")


def main():
    for split in ["train", "validation", "test"]:
        build_split_pkl(split)


if __name__ == "__main__":
    main()
