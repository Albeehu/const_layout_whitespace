#確認高品質layout的樣子
import pickle
import numpy as np
import os
import sys
from PIL import Image, ImageDraw
from tqdm import tqdm

# === Numpy 結構相容性修正 ===
import numpy.core.multiarray
sys.modules['numpy._core.numeric'] = np.core.numeric
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

# ========= 1. 核心參數設定 =========
PKL_PATH = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_all.pkl" 
N_TARGET = 6  # 目標元素數量
SVG_ID, TEXT_ID, IMG_ID = 0, 1, 2

# --- 篩選指標 ---
MIN_WHITESPACE = 0.50      # 留白 50% 以上
MIN_ELEMENT_AREA = 0.005   # 防止物件縮小到消失
MAX_ASPECT_RATIO = 8.0     # 比例保護
MAX_OVERLAP_RATIO = 0.05   # 元素間幾乎不重疊

def select_nodes_v17(bbox, label, n_target=6):
    """
    1. 優先確保各有一個 Img, Text, SVG
    2. 若不足，依優先級 (Img > Text > SVG) 補齊
    3. 總數至少需 5 個
    """
    label = np.array(label)
    idxs = np.arange(len(label))
    
    # 過濾出可用的排版元件
    v_mask = (label == IMG_ID) | (label == TEXT_ID) | (label == SVG_ID)
    v_idxs = idxs[v_mask]
    
    if len(v_idxs) < 5: return None # 至少要 5 個元素

    selected = []
    pool = list(v_idxs)
    
    # 第一階段：優先各拿一個 (Img, Text, SVG)
    for tid in [IMG_ID, TEXT_ID, SVG_ID]:
        cand = [i for i in pool if label[i] == tid]
        if cand:
            p = cand[0] # 取第一個
            selected.append(p)
            pool.remove(p)

    # 第二階段：依優先級補齊剩餘名額 (Image > Text > SVG)
    priority_map = {IMG_ID: 0, TEXT_ID: 1, SVG_ID: 2}
    pool.sort(key=lambda i: priority_map.get(label[i], 3))
    
    needed = n_target - len(selected)
    if needed > 0:
        selected.extend(pool[:needed])
    
    # 最終確認：如果 selected 裡面的元素不夠 5 個 (極端情況)，則捨棄
    if len(selected) < 5: return None
    
    return np.sort(selected)

def calculate_style_scores(bbox):
    """計算留白風格得分"""
    x1, y1 = np.clip(bbox[:, 0]-bbox[:, 2]/2, 0, 1), np.clip(bbox[:, 1]-bbox[:, 3]/2, 0, 1)
    x2, y2 = np.clip(bbox[:, 0]+bbox[:, 2]/2, 0, 1), np.clip(bbox[:, 1]+bbox[:, 3]/2, 0, 1)
    L, R, T, Bm = x1.min(), 1.0-x2.max(), y1.min(), 1.0-y2.max()
    m = np.array([L, R, T, Bm])
    
    s_frame = np.mean(m) - 0.5 * np.std(m)
    s_side = max(L, R) + 0.2 * np.mean([T, Bm])
    s_tb = max(T, Bm) + 0.2 * np.mean([L, R])
    s_hybrid = np.sqrt(max(0, s_side) * max(0, s_tb))
    return [s_frame, s_side, s_tb, s_hybrid]

def is_aesthetic_failure(bbox, label, selected_idxs):
    """美學過濾：留白、面積、比例、重疊"""
    elems = bbox[selected_idxs]
    areas = elems[:, 2] * elems[:, 3]
    
    # 1. 留白檢查
    if (1.0 - np.sum(areas)) < MIN_WHITESPACE: return True
    # 2. 面積保護
    if np.any(areas < MIN_ELEMENT_AREA): return True
    # 3. 比例保護
    ratios = np.maximum(elems[:, 2]/elems[:, 3], elems[:, 3]/elems[:, 2])
    if np.any(ratios > MAX_ASPECT_RATIO): return True
    
    # 4. 重疊檢查
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            bi, bj = elems[i], elems[j]
            ix = max(0, min(bi[0]+bi[2]/2, bj[0]+bj[2]/2) - max(bi[0]-bi[2]/2, bj[0]-bj[2]/2))
            iy = max(0, min(bi[1]+bi[3]/2, bj[1]+bj[3]/2) - max(bi[1]-bi[3]/2, bj[1]-bi[3]/2))
            if (ix * iy) / min(bi[2]*bi[3], bj[2]*bj[3]) > MAX_OVERLAP_RATIO: return True
            
    return False

def draw_layout_v17(all_bbox, all_label, selected_idxs, filename):
    img = Image.new('RGB', (500, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    colors = {0:(31,119,180), 1:(44,160,44), 2:(148,103,189)}
    for i in selected_idxs:
        lbl = int(all_label[i])
        if lbl in colors:
            cx, cy, w, h = all_bbox[i]
            x1, y1, x2, y2 = (cx-w/2)*500, (cy-h/2)*500, (cx+w/2)*500, (cy+h/2)*500
            draw.rectangle([x1, y1, x2, y2], outline=colors[lbl], width=3)
    img.save(filename)

if __name__ == "__main__":
    if not os.path.exists(PKL_PATH):
        print(f"File not found: {PKL_PATH}"); sys.exit()
    
    with open(PKL_PATH, 'rb') as f: data = pickle.load(f)
    style_buckets = [[], [], [], []]
    style_names = ["Frame", "Side", "TopBottom", "Hybrid"]
    
    print(f"啟動全方位多樣化篩選 (N>=5, 優先 Img/Text/Svg)...")
    for i, item in enumerate(tqdm(data)):
        if not item: continue
        bbox, label = np.array(item[0]), np.array(item[1])
        
        sel = select_nodes_v17(bbox, label, n_target=N_TARGET)
        if sel is None: continue
        
        if is_aesthetic_failure(bbox, label, sel): continue
        
        scores = calculate_style_scores(bbox[sel])
        for s_idx in range(4):
            style_buckets[s_idx].append({'idx': i, 'score': scores[s_idx], 'bbox': bbox, 'label': label, 'sel': sel})

    os.makedirs("audit_v17", exist_ok=True)
    all_high_quality_indices = []
    
    for s_idx, name in enumerate(style_names):
        os.makedirs(f"audit_v17/{name}", exist_ok=True)
        bucket = sorted(style_buckets[s_idx], key=lambda x: x['score'], reverse=True)
        # 每個風格取前 150 名以保證多樣性
        top_samples = bucket[:150]
        for rank, sample in enumerate(top_samples):
            all_high_quality_indices.append(sample['idx'])
            if rank < 50: # 視覺化前 50 張
                draw_layout_v17(sample['bbox'], sample[ 'label'], sample['sel'], f"audit_v17/{name}/rank_{rank}.png")
        print(f"風格 {name}: 成功撈取 {len(top_samples)} 筆。")

    # 儲存高品質索引清單，供 v45.py 使用
    final_indices = list(set(all_high_quality_indices))
    with open("high_quality_indices.pkl", 'wb') as f:
        pickle.dump(final_indices, f)
    print(f"\n[完成] 總計獲得 {len(final_indices)} 筆高品質多樣化樣本，索引已存至 high_quality_indices.pkl")