"""生成o.6留白且還有face的pkl 或是生成o.5留白且還有face的pkl 
（留白大小可以用ws_thr 設定）
用法如下
python make_ws0_5_with_face_pkl.py \
  --imgmap_pkl /home/albee/const_layout_whitespace/data/dataset/crello/crello_train_imgmap.pkl \
  --face_pkl   /home/albee/const_layout_whitespace/data/dataset/crello/crello_train_face.pkl \
  --out_pkl    /home/albee/const_layout_whitespace/data/dataset/crello/crello_train_ws_gt0.5_with_face.pkl \
  --ws_thr 0.5 --max_faces 4 --require_face 0
"""
import os
import sys
import pickle
import numpy as np
import argparse

# 讓同資料夾下的 whitespace_metric.py 一定能被 import
sys.path.append(os.path.dirname(__file__))
from whitespace_metric import whitespace_quality  # 你的留白指標

SVG_ID  = 0
TEXT_ID = 1
IMG_ID  = 2
BG_ID   = 3
MASK_ID = 4
FACE_ID = 5

def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N,4) cx,cy,w,h -> (N,4) x1,y1,x2,y2，並 clamp 到 [0,1]"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = np.clip(cx - w / 2.0, 0.0, 1.0)
    y1 = np.clip(cy - h / 2.0, 0.0, 1.0)
    x2 = np.clip(cx + w / 2.0, 0.0, 1.0)
    y2 = np.clip(cy + h / 2.0, 0.0, 1.0)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

def compute_wr_by_whitespace_metric(boxes_cxcywh: np.ndarray,
                                   labels: np.ndarray,
                                   height: int,
                                   width: int,
                                   alpha: float,
                                   exclude_bg: bool = True,
                                   exclude_mask: bool = True,
                                   exclude_face: bool = True) -> float:
    """
    用 whitespace_metric.whitespace_quality() 的同一套邏輯計算 WR。
    注意：whitespace_quality 吃的是 xyxy。
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    keep = np.ones_like(labels, dtype=bool)
    if exclude_bg:
        keep &= (labels != BG_ID)
    if exclude_mask:
        keep &= (labels != MASK_ID)
    if exclude_face:
        keep &= (labels != FACE_ID)

    boxes_xyxy = cxcywh_to_xyxy(np.asarray(boxes_cxcywh, dtype=np.float32)[keep])

    m = whitespace_quality(
        boxes_xyxy,
        height=height,
        width=width,
        alpha=alpha,
    )
    return float(m.get("WR", 0.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgmap_pkl", required=True, help="crello_*_imgmap.pkl (list of (boxes_cxcywh, labels))")
    ap.add_argument("--face_pkl", required=True, help="crello_*_face.pkl (list of face boxes per sample, cxcywh normalized)")
    ap.add_argument("--out_pkl", required=True)
    ap.add_argument("--ws_thr", type=float, default=0.5)
    ap.add_argument("--max_faces", type=int, default=4)
    ap.add_argument("--require_face", type=int, default=0, help="1=只保留有 face 的樣本；0=沒 face 也保留")

    # 跟 whitespace_metric.py 對齊的參數
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.1)

    # ws 計算時要不要排除 BG/MASK/FACE（預設都排除）
    ap.add_argument("--ws_exclude_bg", type=int, default=1)
    ap.add_argument("--ws_exclude_mask", type=int, default=1)
    ap.add_argument("--ws_exclude_face", type=int, default=1)

    args = ap.parse_args()

    data = pickle.load(open(args.imgmap_pkl, "rb"))   # list[(boxes, labels)]
    faces = pickle.load(open(args.face_pkl, "rb"))    # list[list[[cx,cy,w,h], ...]]

    assert len(data) == len(faces), f"len(imgmap)={len(data)} != len(face)={len(faces)}"

    out = []
    kept = 0
    dropped_no_face = 0
    dropped_ws = 0
    face_appended = 0

    for (boxes, labels), face_list in zip(data, faces):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if face_list is None:
            face_list = []

        if args.require_face == 1 and len(face_list) == 0:
            dropped_no_face += 1
            continue

        # append face（最多 max_faces）
        if len(face_list) > 0:
            f = np.asarray(face_list, dtype=np.float32).reshape(-1, 4)[:args.max_faces]
            fl = np.full((f.shape[0],), FACE_ID, dtype=np.int64)
            boxes2 = np.concatenate([boxes, f], axis=0)
            labels2 = np.concatenate([labels, fl], axis=0)
            face_appended += int(f.shape[0])
        else:
            boxes2, labels2 = boxes, labels

        # 用你的 whitespace_metric 同邏輯算 WR
        wr = compute_wr_by_whitespace_metric(
            boxes2, labels2,
            height=args.height, width=args.width, alpha=args.alpha,
            exclude_bg=bool(args.ws_exclude_bg),
            exclude_mask=bool(args.ws_exclude_mask),
            exclude_face=bool(args.ws_exclude_face),
        )

        if wr < args.ws_thr:
            dropped_ws += 1
            continue

        out.append((boxes2.astype(np.float32), labels2.astype(np.int64)))
        kept += 1

    pickle.dump(out, open(args.out_pkl, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("saved:", args.out_pkl)
    print("kept:", kept)
    print("dropped_no_face:", dropped_no_face)
    print("dropped_ws:", dropped_ws)
    print("face_appended_total:", face_appended)

if __name__ == "__main__":
    main()