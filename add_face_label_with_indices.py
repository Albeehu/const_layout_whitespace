#生成只有face label的pkl

#!/usr/bin/env python
"""
用 imgmap_indices + YOLOv8-face 幫 Crello 每張圖偵測人臉，
輸出一個 face_boxes.pkl：

  face_boxes[i] = list of [cx, cy, w, h]  (0~1)

完全不動原本的 imgmap 結構。

使用方式（例）：
  python build_crello_face_boxes.py \
    --imgmap data/dataset/crello/crello_test_imgmap.pkl \
    --indices data/dataset/crello/crello_test_imgmap_indices.pkl \
    --img-root /home/albee/const_layout_test/crello_v4_test_images \
    --model /home/albee/const_layout_test/yolov8n-face.pt \
    --out data/dataset/crello/crello_test_face_boxes.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# 如果 indices 裡是 dict，這裡是你「圖片路徑」的 key 名稱
# 你可以之後改成 "img_path" / "png_path" 之類
INDEX_IMG_KEY = "image_path"


def detect_faces_xywh01(model, img_path: str, conf: float = 0.25):
    """
    對單張圖做人臉偵測，回傳 list[[cx, cy, w, h], ...]，範圍都是 0~1。
    如果沒臉就回傳 []。
    """
    results = model(img_path, conf=conf, verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N, 4)
    h, w = r.orig_shape  # 高、寬

    faces = []
    for x1, y1, x2, y2 in boxes_xyxy:
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        faces.append([float(cx), float(cy), float(bw), float(bh)])

    return faces


def resolve_image_path_from_index(idx_entry, img_root: Path, i: int) -> Path:
    """
    根據 indices 裡的 entry 推出圖片路徑。

    - 如果 entry 是字串：直接當成路徑
    - 如果是 dict：用 INDEX_IMG_KEY，或常見幾個備援 key
    - 如果都拿不到，就 fallback 成 img_root / f"{i:06d}.png"
    """
    # 1) entry 是字串：直接當成路徑
    if isinstance(idx_entry, str):
        p = Path(idx_entry)
        if p.is_absolute():
            return p
        return img_root / p

    # 2) entry 是 dict：用指定的 key 取路徑
    if isinstance(idx_entry, dict):
        if INDEX_IMG_KEY in idx_entry:
            p = Path(str(idx_entry[INDEX_IMG_KEY]))
            if p.is_absolute():
                return p
            return img_root / p

        # 備用 key（你之後可以依實際情況加/改）
        for k in ["img_path", "path", "file", "filename", "png_path"]:
            if k in idx_entry:
                p = Path(str(idx_entry[k]))
                if p.is_absolute():
                    return p
                return img_root / p

    # 3) fallback：假設檔名是 000000.png 這種連號
    return img_root / f"{i:06d}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgmap", type=str, required=True,
                        help="原本 Crello imgmap pkl 路徑 (只用來取長度)")
    parser.add_argument("--indices", type=str, required=True,
                        help="imgmap_indices pkl 路徑")
    parser.add_argument("--img-root", type=str, required=True,
                        help="圖片所在根目錄，例如 crello_v4_test_images")
    parser.add_argument("--model", type=str, required=True,
                        help="YOLOv8-face 權重路徑 (e.g. yolov8n-face.pt)")
    parser.add_argument("--out", type=str, required=True,
                        help="輸出的 face_boxes pkl 檔名")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="人臉偵測 confidence threshold")
    args = parser.parse_args()

    imgmap_path = Path(args.imgmap)
    indices_path = Path(args.indices)
    img_root = Path(args.img_root)
    out_path = Path(args.out)

    print(f"[INFO] 讀取 imgmap: {imgmap_path}")
    with open(imgmap_path, "rb") as f:
        imgmap = pickle.load(f)

    print(f"[INFO] 讀取 indices: {indices_path}")
    with open(indices_path, "rb") as f:
        indices_obj = pickle.load(f)

    if isinstance(imgmap, list):
        n_imgmap = len(imgmap)
    else:
        try:
            n_imgmap = len(imgmap)
        except TypeError:
            raise TypeError(f"imgmap 型別 {type(imgmap)} 不支援 len()，請看一下結構。")

    # 把 indices 變成 list
    if isinstance(indices_obj, list):
        indices_list = indices_obj
    elif isinstance(indices_obj, dict):
        for k in ["items", "data", "indices", "layouts"]:
            if k in indices_obj and isinstance(indices_obj[k], list):
                indices_list = indices_obj[k]
                break
        else:
            raise TypeError("indices 是 dict，但找不到存 list 的 key（items/data/indices/layouts），請看一下 pkl 結構。")
    else:
        raise TypeError(f"不支援的 indices 類型: {type(indices_obj)}")

    if n_imgmap != len(indices_list):
        print(f"[WARN] imgmap 長度 {n_imgmap} 和 indices 長度 {len(indices_list)} 不同，將以 min(len) 為準。")
        n = min(n_imgmap, len(indices_list))
    else:
        n = n_imgmap

    print(f"[INFO] 將為前 {n} 筆資料建立 face_boxes。")

    print(f"[INFO] 載入人臉模型: {args.model}")
    model = YOLO(args.model)

    face_boxes_all = []  # list of list[[cx, cy, w, h]]

    for i in tqdm(range(n)):
        idx_entry = indices_list[i]
        img_path = resolve_image_path_from_index(idx_entry, img_root, i)

        if not img_path.exists():
            print(f"[WARN] 圖片不存在，face_boxes 設為空：{img_path}")
            faces = []
        else:
            faces = detect_faces_xywh01(model, str(img_path), conf=args.conf)

        face_boxes_all.append(faces)

    print(f"[INFO] 寫出 face_boxes 到: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(face_boxes_all, f)

    print("[DONE] 完成！face_boxes[i] 對應 imgmap[i]。")


if __name__ == "__main__":
    main()
