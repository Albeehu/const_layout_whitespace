#計算留白的指標2025/11/22
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
whitespace_metric.py

Compute whitespace-related metrics for design layouts.

Each layout is represented as a list of bounding boxes:
    boxes = [
        [x1, y1, x2, y2],
        ...
    ]
where coordinates are normalized to [0, 1] relative to canvas width/height.
"""

from __future__ import annotations #開啟較新版本的型別註解寫法（在舊版 Python 也能用）

import argparse #argparse：處理 command line 參數，例如 --input、--output
import csv #結果寫成csv檔
import pickle #read .pkl
from pathlib import Path #檔案路徑物件
from typing import Dict, List, Tuple #typing：型別註解用，不影響執行

import numpy as np #numpy：做矩陣運算、mask
from scipy.ndimage import label  # pip install scipy
#scipy.ndimage.label：做連通元件標記（connected components），用來找留白區塊


# -----------------------------
# Core functions
# -----------------------------
#把 bounding boxes 轉成 2D 佔用圖
#定義一個 function，輸入是一組 boxes，輸出是 height × width 的 2D mask
def layout_to_mask(
    boxes: np.ndarray,
    height: int = 128,
    width: int = 128,
    large_thresh: float = 0.95,
) -> np.ndarray:
    """
    Convert a list/array of bounding boxes into a binary occupancy mask.

    Args:
        boxes: array of shape (N, 4) with [x1, y1, x2, y2] in [0, 1].
        height: number of rows of the mask.
        width: number of columns of the mask.

    Returns:
        mask: np.ndarray of shape (height, width), dtype=uint8
              1 = occupied (content), 0 = empty (potential whitespace).
    """
    #mask就會是128*128且ele.都是0的matrix
    # Start with all zeros: assume everything is empty (whitespace).
    mask = np.zeros((height, width), dtype=np.uint8)

    #layout沒有box return 全留白(0)的畫布
    if boxes is None or len(boxes) == 0:
        return mask

    # Ensure boxes is a numpy array of shape (N, 4).用-1讓np自己算有幾列
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    for (x1, y1, x2, y2) in boxes:
        # 先算這個 box 在 normalized 座標下的寬高/面積
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh

        #如果你定義的是「面積占比 >= 0.95 就忽略」
        if area >= large_thresh:
            continue

    #把每個box取出處理 np.clip(要處理的數,下限,上限)
    for (x1, y1, x2, y2) in boxes:
        # Clamp to [0, 1] just in case.
        x1 = np.clip(x1, 0.0, 1.0)
        y1 = np.clip(y1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        y2 = np.clip(y2, 0.0, 1.0)

        # Convert normalized coordinates to integer pixel indices.
        #把 normalized 座標乘上寬高轉成 pixel index
        #右下角用 ceil（向外取整），確保不會漏掉邊界的一列 / 一行
        x1p = int(x1 * width)
        x2p = int(np.ceil(x2 * width))
        y1p = int(y1 * height)
        y2p = int(np.ceil(y2 * height))

        # Safety: ensure indices are within bounds.
        #再次clamp確保不超出
        x1p = np.clip(x1p, 0, width)
        x2p = np.clip(x2p, 0, width)
        y1p = np.clip(y1p, 0, height)
        y2p = np.clip(y2p, 0, height)

        #避免奇怪的反向框（x2 < x1）或零寬 / 零高框，直接跳過
        if x2p <= x1p or y2p <= y1p:
            # Degenerate box, skip.
            continue
        
        #在 mask 上把這個 box 範圍標記成 1，代表「有內容」（不是留白）
        # Mark region as occupied (content).
        mask[y1p:y2p, x1p:x2p] = 1

    return mask


#直接從留白 mask 算指標
#alpha 控制「留白碎片數量」的懲罰強度
def whitespace_metrics_from_mask(
    whitespace_mask: np.ndarray, #型別注記是np.array
    alpha: float = 0.1,
) -> Dict[str, float]: #輸出一組指標
    """
    Compute whitespace-related metrics given a binary whitespace mask.

    Args:
        whitespace_mask: 2D array, 1 = whitespace, 0 = occupied.
        alpha: fragmentation penalty strength.

    Returns:
        dict with keys: WR, LWR, DWS, num_cc.
    """
    h, w = whitespace_mask.shape #取得mask寬高
    total_pixels = float(h * w) #總像素

    # Global whitespace ratio.全局留白比例
    #whitespace_mask.sum() 是留白 pixel 數
    wr = float(whitespace_mask.sum()) / total_pixels

    #如果幾乎沒有留白，直接回傳全部 0（除了 WR）
    if wr < 1e-8:
        # No whitespace at all.
        return dict(WR=wr, LWR=0.0, DWS=0.0, num_cc=0,
            cx=0.5, cy=0.5,
            S_frame=0.0, S_side=0.0, S_tb=0.0, S_pos=0.0
        )

    # Connected components on whitespace.
    # 對留白 mask 做 connected component labeling，第二塊留白上面就會標2...
    #labeled：每個 pixel 上寫 component id
    labeled, num_cc = label(whitespace_mask)
    #理論上不會發生（因為 wr>0 應該就有元件），但保險一下
    if num_cc == 0:
        # Should not happen if wr > 0, but handle for safety.
        return dict(WR=wr, LWR=0.0, DWS=0.0, num_cc=0,
            cx=0.5, cy=0.5,
            S_frame=0.0, S_side=0.0, S_tb=0.0, S_pos=0.0
        )


    # Compute area of each connected component (skip label 0 = background).
    #np.bincount 計算每個 label（0,1,2,...）的像素數
    #label 0 是背景（非留白），切掉，只留真正的留白區塊
    areas = np.bincount(labeled.ravel())
    areas = areas[1:]  # drop background

    largest_area = float(areas.max())
    lwr = largest_area / total_pixels #最大留白連通面積

    # Dominant whitespace vs total whitespace.
    #所有留白裡面，有多少比例集中在最大那一塊，1e-8避免除以0
    dws_conn = lwr / (wr + 1e-8)

     # 3) 留白重心 (cx, cy) whitespace_mask 1=留白,0=有ele.
    ys, xs = np.nonzero(whitespace_mask) #nonzero找出非0 ele.位置
    if len(xs) == 0: #完全沒留白
        cx = cy = 0.5
    else:
        #xs.mean()算xs的平均
        cx = xs.mean() / float(w)
        cy = ys.mean() / float(h)

    # Fragmentation penalty: more components -> smaller penalty.
    #碎片懲罰：如果只有 1 塊留白 → penalty = 1
    #元件多 → denominator 變大 → penalty 變小
    frag_penalty = 1.0 / (1.0 + alpha * float(num_cc - 1))
    #最終留白品質指標 DWS = 集中度 × 碎片懲罰
    # 新版 DWS：連通性 × 位置樣式（× 碎片懲罰）
    # DWS_new = dws_conn * frag_penalty * S_pos
    return dict(
    WR=wr,
    LWR=lwr,
    num_cc=float(num_cc),
    cx=cx,
    cy=cy,
    )   
    # return dict(
    #     WR=wr,
    #     LWR=lwr,
    #     DWS=DWS_new,
    #     num_cc=float(num_cc),
    #     cx=cx,
    #     cy=cy,
    #     S_frame=S_frame,
    #     S_side=S_side,
    #     S_tb=S_tb,
    #     S_pos=S_pos,
    # )

def content_margins_from_boxes(boxes: np.ndarray):
    """
    根據內容框算出四周邊距（normalized）：
        L = 左邊留白寬度
        R = 右邊留白寬度
        T = 上方留白高度
        B = 下方留白高度
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    if boxes.size == 0:
        # 沒有內容 → 當作四邊都滿留白
        return 1.0, 1.0, 1.0, 1.0

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # ★ 先 clamp 到 [0,1]，跟 layout_to_mask 的邏輯一致
    x1 = np.clip(x1, 0.0, 1.0)
    y1 = np.clip(y1, 0.0, 1.0)
    x2 = np.clip(x2, 0.0, 1.0)
    y2 = np.clip(y2, 0.0, 1.0)

    x1_min = float(x1.min())
    y1_min = float(y1.min())
    x2_max = float(x2.max())
    y2_max = float(y2.max())

    L = x1_min          # 左：內容最左邊到畫布左邊的距離
    R = 1.0 - x2_max    # 右：內容最右邊到畫布右邊的距離
    T = y1_min          # 上：內容最上方到頂邊
    B = 1.0 - y2_max    # 下：內容最下方到底邊

    # 再保險一次，確保在 [0,1]
    L = float(np.clip(L, 0.0, 1.0))
    R = float(np.clip(R, 0.0, 1.0))
    T = float(np.clip(T, 0.0, 1.0))
    B = float(np.clip(B, 0.0, 1.0))

    return L, R, T, B

def style_scores_from_margins(
    L: float, R: float, T: float, B: float
) -> Dict[str, float]:
    """
    根據四周邊距算三種 style 的分數：
        frame : 四周都留白、邊距大又平均
        side  : 左右有一側留白特別大
        tb    : 上下有一側留白特別大
    """
    margins = np.array([L, R, T, B], dtype=np.float32)
    mean_m = float(margins.mean())
    std_m = float(margins.std()) #取margins內數值的標準差，數值差不多->std小，四周平均 數值差很多->std大則留白不均

    # 四周留白：邊距越大越好，越平均越好(平均扣掉不平均的懲罰)
    S_frame = mean_m - 0.5 * std_m

    # 側邊留白：左右其中一側特別大
    horiz = np.array([L, R], dtype=np.float32)
    h_max = float(horiz.max())
    h_min = float(horiz.min())

    vert = np.array([T, B], dtype=np.float32)
    v_mean = float(vert.mean())

    # 一側大(h_max)、左右差大(h_max-h_min)，加上一點上下空間
    S_side = h_max + 0.8 * (h_max - h_min) + 0.2 * v_mean

    # 上下留白：上下其中一側特別大
    vert = np.array([T, B], dtype=np.float32)
    v_max = float(vert.max())
    v_min = float(vert.min())

    h_mean = float(horiz.mean())

    S_tb = v_max + 0.8 * (v_max - v_min) + 0.2 * h_mean

    # clip 到 [0,1]，避免出現負數或 >1
    S_frame = float(np.clip(S_frame, 0.0, 1.0))
    S_side  = float(np.clip(S_side, 0.0, 1.0))
    S_tb    = float(np.clip(S_tb,   0.0, 1.0))

    # 這個當作「主風格」的 summary（保留給 debug / 分析用）
    S_pos = max(S_frame, S_side, S_tb)

    # 新增一個「綜合 style 分數」：三個一起看
    # corner 留白（side + tb 都大）就會拿到比只大一個更高的分數
    w_frame = 0.5   # 四周留白也 ok，但不是主角
    w_side  = 1.0   # 側邊留白加權高一點
    w_tb    = 1.0   # 上下留白加權高一點

    S_style = (
        w_frame * S_frame +
        w_side  * S_side  +
        w_tb    * S_tb
    ) / (w_frame + w_side + w_tb)


    return dict(S_frame=S_frame, S_side=S_side, S_tb=S_tb, S_pos=S_pos, S_style=S_style,)


#輸入bounding boxes，會呼叫layout_to_mask + whitespace_metrics_from_mask
def whitespace_quality(
    boxes: np.ndarray,
    height: int = 128,
    width: int = 128,
    alpha: float = 0.1,
) -> Dict[str, float]:
    """
    Convenience wrapper: from bounding boxes to whitespace metrics.

    Args:
        boxes: array-like of shape (N, 4) in normalized coordinates.
        height: mask height (discretization resolution).
        width: mask width.
        alpha: fragmentation penalty parameter.

    Returns:
        dict with keys: WR, LWR, DWS, num_cc.
    """
    #先算出「佔用 mask」，再取反變成「留白 mask」
    occ_mask = layout_to_mask(boxes, height=height, width=width)
    whitespace_mask = 1 - occ_mask  # 1 = whitespace, 0 = content
    #丟給前一個函式算 WR / LWR / DWS / num_cc
    # 2) 先算 WR / LWR / num_cc（連通性部分）
    #1219 S_side + S_tb一起算
    m = whitespace_metrics_from_mask(whitespace_mask, alpha=alpha)
    wr = m["WR"]
    lwr = m["LWR"]
    num_cc = m.get("num_cc", 1.0)

    # 3) 用內容框算四周邊距 + style 分數
    L, R, T, B = content_margins_from_boxes(boxes)
    style = style_scores_from_margins(L, R, T, B)

    # 4) D_conn：最大留白 / 全部留白（留白集中度）
    if wr > 1e-8:
        D_conn = lwr / (wr + 1e-8)
    else:
        D_conn = 0.0

    # 5) 碎片懲罰：留白被切的越多塊，分數越低
    #    （如果你的 num_cc 幾乎都 1，這項 ≈ 1，不太影響）
    frag_penalty = 1.0 / (1.0 + alpha * max(0.0, float(num_cc) - 1.0))

    # 6) style 綜合分數：同時看 frame / side / tb
    #    corner 留白（側邊 + 上下都大）會讓 S_side 和 S_tb 一起變大 → S_style 特別高
    S_style = style["S_style"]   # 注意：style_scores_from_margins 要有回傳 S_style

    # 7) 新版 DWS：留白集中度 × 碎片懲罰 × 留白風格分數
    DWS_new = D_conn * frag_penalty * S_style
    m["DWS"] = DWS_new

    # 8) 把補充資訊也放進 metric dict，方便你在 CSV 裡看
    m.update(dict(
        L=L, R=R, T=T, B=B,
        S_frame=style["S_frame"],
        S_side=style["S_side"],
        S_tb=style["S_tb"],
        S_pos=style["S_pos"],      # 主風格（max），拿來分類用
        S_style=S_style,           # 綜合 style 分數，用來算 DWS
    ))

    return m
    #1219 end

# -----------------------------
# I/O helpers (you may need to adapt to your format)
# -----------------------------
#定義一個從 pickle 檔讀 layout 的 helper
def load_layouts_from_pkl(path: Path) -> List[np.ndarray]:
    """
    Load layouts from a pickle file.

    支援 LayoutGAN++ (const_layout) 的 generated.pkl 格式：
        layouts = [
            (boxes_xywh, labels, ...),
            ...
        ]
    其中 boxes_xywh 是 (N, 4) 的 array，格式 [x, y, width, height]，座標已經是 [0,1] normalized。

    回傳:
        layouts: List[np.ndarray]，每個元素是 (N, 4) 的 [x1, y1, x2, y2] 陣列。
    """
    with path.open("rb") as f:
        data = pickle.load(f)

    layouts: List[np.ndarray] = []

    # ---- Case 0: LayoutGAN++ / const_layout 官方格式 ----
    # data 是 list，裡面每個元素是 tuple，第一個元素是 boxes_xywh
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
        for layout in data:
            # layout[0] 應該是 (N, 4) 的 boxes_xywh
            boxes_xywh = np.asarray(layout[0], dtype=np.float32).reshape(-1, 4)

            # 拆成 x, y, w, h
            x = boxes_xywh[:, 0]
            y = boxes_xywh[:, 1]
            w = boxes_xywh[:, 2]
            h = boxes_xywh[:, 3]

            # 轉成 [x1, y1, x2, y2]，方便後面做遮罩
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)
            layouts.append(boxes_xyxy)

        return layouts

    # ---- Case 1: data 是 list，裡面直接是 (N,4) 的 array/list ----
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
        try:
            arr0 = np.asarray(data[0])
        except Exception:
            arr0 = None

        if arr0 is not None and arr0.ndim == 2 and arr0.shape[1] >= 4:
            for boxes in data:
                boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                layouts.append(boxes_arr)
            return layouts

    # ---- Case 2: data 是 list of dict，且有 "bboxes" key ----
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        for item in data:
            if "bboxes" in item:
                boxes = np.asarray(item["bboxes"], dtype=np.float32).reshape(-1, 4)
                layouts.append(boxes)
            else:
                layouts.append(np.zeros((0, 4), dtype=np.float32))
        return layouts

    # ---- Fallback: 把整個 data 當成一張 layout ----
    arr = np.asarray(data, dtype=np.float32).reshape(-1, 4)
    layouts.append(arr)
    return layouts


#輸入是每張 layout 的 metric dict list，輸出寫到 CSV 檔
def save_metrics_to_csv(
    metrics_list: List[Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Save list of metric dicts to a CSV file.

    Each row corresponds to one layout.
    """
    #如果 list 空的，直接回報沒有東西
    if not metrics_list:
        print("No metrics to save.")
        return
    #定義 CSV 欄位名稱
    fieldnames = ["index", "WR", "LWR", "DWS", "num_cc","L", "R", "T", "B",
    "S_frame", "S_side", "S_tb", "S_pos", "S_style",]
    #開啟檔案（寫入模式、UTF-8），建立一個 DictWriter，先寫表頭
    with output_path.open("w", newline="", encoding="utf-8") as f:
        #DictWriter專門用來把 字典 dict 寫成 CSV
        #一個 list，裡面是欄位名稱的順序
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #先把欄位名稱那一列寫進 CSV 的第一行(fieldnames)
        writer.writeheader()
        #DictWriter 會按照 fieldnames 的順序輸出欄位，不會因為 dict 是無序的就亂掉
        for idx, m in enumerate(metrics_list):
            row = {
                "index": idx,
                "WR": m.get("WR", 0.0),
                "LWR": m.get("LWR", 0.0),
                "DWS": m.get("DWS", 0.0),
                "num_cc": m.get("num_cc", 0.0),
                "L": m.get("L", 0.0),
                "R": m.get("R", 0.0),
                "T": m.get("T", 0.0),
                "B": m.get("B", 0.0),
                "S_frame": m.get("S_frame", 0.0),
                "S_side":  m.get("S_side", 0.0),
                "S_tb":    m.get("S_tb", 0.0),
                "S_pos":   m.get("S_pos", 0.0),
                "S_style":   m.get("S_style", 0.0),
            }
            #wirte對應的值
            writer.writerow(row)


# -----------------------------
# Command-line interface
# -----------------------------
#def command line 參數
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    #建一個 parser，並寫一段簡短說明
    parser = argparse.ArgumentParser(
        description="Compute whitespace metrics for layouts."
    )
    #定義 --input 參數（必填），用來指定 .pkl 檔路徑
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input pickle file containing layouts.",
    )
    #定義 --output 參數（必填），輸出 .csv 路徑
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file for metrics.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Mask height (discretization resolution).",
    )
    #--height：mask 的高度（預設 128）
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Mask width (discretization resolution).",
    )
    #--width：mask 的寬度（預設 128）
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Fragmentation penalty parameter.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point when run as a script.
    """
    #讀取命令列參數
    args = parse_args()
    #把字串路徑包成 Path 物件
    input_path = Path(args.input)
    output_path = Path(args.output)

    #印出正在讀的檔案，呼叫 load_layouts_from_pkl，並顯示一共載了幾張 layout
    print(f"[INFO] Loading layouts from: {input_path}")
    layouts = load_layouts_from_pkl(input_path)
    print(f"[INFO] Loaded {len(layouts)} layouts.")

    #建一個空 list，準備把每張 layout 的指標收進去
    metrics_list: List[Dict[str, float]] = []

    #逐張 layout loop：call whitespace_quality算出 WR、LWR、DWS、num_cc
    for idx, boxes in enumerate(layouts):
        m = whitespace_quality(
            boxes,
            height=args.height,
            width=args.width,
            alpha=args.alpha,
        )
        #append到list
        metrics_list.append(m)
        #每100張印一次進度
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Processed {idx + 1} layouts...")
    #呼叫 save_metrics_to_csv 把所有指標存成 .csv
    print(f"[INFO] Saving metrics to: {output_path}")
    save_metrics_to_csv(metrics_list, output_path)
    print("[INFO] Done.")

#呼叫 save_metrics_to_csv 把所有指標存成 .csv
if __name__ == "__main__":
    main()
