#新增mainpart這個label
import os
import pickle
import cv2
from ultralytics import YOLO
import numpy as np

# ========= 你要確認 / 修改的設定 =========

NEW_LABEL_ID = 5 

# 這就是你現在這個檔：裡面是 list，每筆是 (boxes, labels)
PKL_IN = "/home/albee/const_layout_test/data/dataset/crello/crello_train_imgmap.pkl"

# Crello test split 的圖片資料夾
# 要跟你當初 render 出來的檔名對得上，假設是 000000.png ~ 002374.png
IMG_DIR = "/home/albee/const_layout_test/crello_v4_train_images"

# 新輸出的檔案：在每筆多加 mainpart
PKL_OUT = "/home/albee/const_layout_test/data/dataset/crello/crello_train_imgmap_with_mainpart.pkl"

# 哪些 layout label 代表 ImageElement（要依你自己的 mapping 改）
IMAGE_CLASS_IDS = [2]  # 例如 2 是 image / 背景那種 element

# YOLO 裡哪些類別算「主體」
# 0 = COCO 的 person；如果要加產品類可以改成 [0, 39, 41, 67] 之類
ALLOWED_MAINPART_CLASSES = [0]

# =======================================


def denorm_box(box, W, H):
    """[cx, cy, w, h] (0~1) → [x1, y1, x2, y2] pixel"""
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))
    return [x1, y1, x2, y2]


def intersect_box(det_box, elem_box):
    """兩個 pixel box 的交集，沒有交集回 None"""
    x1 = max(det_box[0], elem_box[0])
    y1 = max(det_box[1], elem_box[1])
    x2 = min(det_box[2], elem_box[2])
    y2 = min(det_box[3], elem_box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def to_norm_box(box, W, H):
    """[x1, y1, x2, y2] pixel → [cx, cy, w, h] (0~1)"""
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return [float(cx), float(cy), float(w), float(h)]


def point_in_box(px, py, box):
    """(px,py) 是否在 box 裡"""
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def main():
    # 1. 讀 crello_test_imgmap.pkl
    with open(PKL_IN, "rb") as f:
        data = pickle.load(f)

    print("樣本數:", len(data))  # 應該是 2375

    # 2. YOLO 模型（用 COCO 預訓練就好）
    model = YOLO("yolov8n.pt")

    new_data = []

    for idx, rec in enumerate(data):
        # rec 應該就是 (boxes, labels)
        if not (isinstance(rec, tuple) and len(rec) == 2):
            print(f"[警告] 第 {idx} 筆不是 (boxes, labels) 的 tuple，直接原樣保留")
            new_data.append(rec)
            continue

        boxes, labels = rec
        img_path = os.path.join(IMG_DIR, f"{idx:06d}.png")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[警告] 找不到圖片: idx={idx}, 路徑={img_path}，這筆 mainpart 全設 None")
            mainpart = [None] * len(boxes)
            new_data.append((boxes, labels, mainpart))
            continue

        H, W = img.shape[:2]

        # 每個 element 的像素框（只有 ImageElement 有）
        element_pixel_boxes = []
        for b, lbl in zip(boxes, labels):
            if int(lbl) in IMAGE_CLASS_IDS:
                element_pixel_boxes.append(denorm_box(b, W, H))
            else:
                element_pixel_boxes.append(None)

        # mainpart[j]：第 j 個 element 的主體資訊（預設 None）
        mainpart = [None] * len(boxes)

        # YOLO 偵測
        results = model(img)[0]

        for det in results.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])

            # 只保留你定義為「主體」的 YOLO 類別
            if ALLOWED_MAINPART_CLASSES and cls_id not in ALLOWED_MAINPART_CLASSES:
                continue

            det_box = [x1, y1, x2, y2]
            cx_det = (x1 + x2) / 2.0
            cy_det = (y1 + y2) / 2.0

            # 指派給某個 ImageElement
            for elem_idx, elem_box in enumerate(element_pixel_boxes):
                if elem_box is None:
                    continue

                if not point_in_box(cx_det, cy_det, elem_box):
                    continue

                # 交集 → 確保主體框不超出 ImageElement 範圍
                sub_box = intersect_box(det_box, elem_box)
                if sub_box is None:
                    continue

                norm_box = to_norm_box(sub_box, W, H)
                candidate = {
                    "bbox": norm_box,   # [cx, cy, w, h] normalized，相對整張圖
                    "class_id": cls_id,
                    "conf": conf,
                }

                # 若尚未有 mainpart 或這個 conf 更高，就更新
                if (mainpart[elem_idx] is None) or (conf > mainpart[elem_idx]["conf"]):
                    mainpart[elem_idx] = candidate

                break  # 一個 detection 只指派給一個 ImageElement

            # -------- 在這裡（for det 迴圈結束之後）把 mainpart 變成 label=5 --------

            n_orig = len(mainpart)  # 原始 element 個數（不包含以前多出來的 5）

            # 只拿「原本的」 boxes / labels 來當基礎
            boxes_list  = [boxes[i].astype(np.float32) for i in range(n_orig)]
            labels_list = [int(labels[i]) for i in range(n_orig)]

            # 對每個原始 element，看 mainpart 有沒有東西
            for mp in mainpart:
                if mp is None:
                    continue
                # mp["bbox"] 是 [cx, cy, w, h]（0~1，相對整張圖）
                b = np.array(mp["bbox"], dtype=np.float32)
                boxes_list.append(b)
                labels_list.append(NEW_LABEL_ID)   # 新的類別 id = 5

            # 轉回 numpy array
            boxes2  = np.stack(boxes_list, axis=0)
            labels2 = np.array(labels_list, dtype=np.int64)

            # 存進 new_data：每筆是 (boxes2, labels2, mainpart)
            new_data.append((boxes2, labels2, mainpart))



        if idx % 50 == 0:
            print(f"已處理 {idx} / {len(data)}")

    # 3. 存回新的 pkl：每筆是 (boxes, labels, mainpart)
    with open(PKL_OUT, "wb") as f:
        pickle.dump(new_data, f)

    print("完成，輸出檔案:", PKL_OUT)


if __name__ == "__main__":
    main()
