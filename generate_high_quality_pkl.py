# 生成高品質的layout pkl
# 生成高品質的layout pkl（輸出 list of (bbox, label)）
import pickle
import numpy as np
import os
import sys
from tqdm import tqdm

# Numpy 相容性修正
import numpy.core.multiarray
sys.modules['numpy._core.numeric'] = np.core.numeric
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

INPUT_PKL = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_all.pkl"   # 使用全集作為篩選來源
OUTPUT_PKL = "/home/albee/const_layout_whitespace/data/dataset/crello/high_quality.pkl"       # 改成真正樣本檔

SVG_ID, TEXT_ID, IMG_ID = 0, 1, 2
MIN_WHITESPACE = 0.50
MAX_OVERLAP = 0.05   # 目前這份程式其實沒真的用到，可先保留

def calculate_align(bbox):
    if len(bbox) < 2:
        return 0.5
    x1, y1 = bbox[:, 0] - bbox[:, 2] / 2, bbox[:, 1] - bbox[:, 3] / 2
    x2, y2 = bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2
    xc, yc = bbox[:, 0], bbox[:, 1]
    c, t = 0, 0
    for i in range(len(bbox)):
        for j in range(i + 1, len(bbox)):
            t += 1
            if min([
                abs(x1[i] - x1[j]),
                abs(x2[i] - x2[j]),
                abs(xc[i] - xc[j]),
                abs(y1[i] - y1[j]),
                abs(y2[i] - y2[j]),
                abs(yc[i] - yc[j]),
            ]) < 0.012:
                c += 1
    return c / t if t > 0 else 0.5

def safe_to_numpy(x, dtype=None):
    arr = np.array(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr

if __name__ == "__main__":
    with open(INPUT_PKL, "rb") as f:
        data = pickle.load(f)

    hq_samples = []

    print("正在分析高品質樣本...")
    for i, item in enumerate(tqdm(data)):
        if item is None:
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        bbox = safe_to_numpy(item[0], dtype=np.float32)
        label = safe_to_numpy(item[1], dtype=np.int64)

        # 基本格式檢查
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            continue
        if label.ndim != 1:
            continue
        if len(bbox) != len(label):
            continue

        v_idxs = [idx for idx, l in enumerate(label) if l in [SVG_ID, TEXT_ID, IMG_ID]]

        # 1. 數量限制：至少 5 個可見元素
        if len(v_idxs) < 5:
            continue

        # 2. 類別多樣性：必須有圖、有文
        unique_l = set(label[v_idxs].tolist())
        if not {IMG_ID, TEXT_ID}.issubset(unique_l):
            continue

        # 3. 留白檢查
        areas = bbox[v_idxs, 2] * bbox[v_idxs, 3]
        whitespace = 1.0 - float(np.sum(areas))
        if whitespace < MIN_WHITESPACE:
            continue

        # 4. 對齊得分
        align = calculate_align(bbox[v_idxs])
        if align <= 0.3:
            continue

        # 收進真正的樣本，而不是索引
        hq_samples.append((bbox, label))

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(hq_samples, f)

    print(f"成功產生 {len(hq_samples)} 筆高品質樣本，已存至 {OUTPUT_PKL}")

    # debug 看第一筆格式
    if len(hq_samples) > 0:
        print("first sample type:", type(hq_samples[0]))
        print("first bbox shape:", hq_samples[0][0].shape)
        print("first label shape:", hq_samples[0][1].shape)