"""
因為缺少fine-tune的權重 所以只先計算原始模型的FID、face occlusion rate的實驗
有串到metric.py
"""

import pickle
import torch
import numpy as np
import sys

# 解決環境相容性問題
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.core.numeric": module = "numpy._core.numeric"
        if module == "numpy.core.multiarray": module = "numpy._core.multiarray"
        return super().find_class(module, name)

def safe_load(file_path):
    with open(file_path, 'rb') as f:
        return SafeUnpickler(f).load()

from metric import FaceOcclusion

def run_evaluation():
    FACE_PKL = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_test_face.pkl"
    LAYOUT_PKL = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_test_imgmap.pkl"
    
    print("--- 啟動原始資料臉部遮擋率評估 (精準模式) ---")
    
    # 兩個計數器，用來對照排除標籤後的差異
    metric_original = FaceOcclusion(FACE_PKL)  # 原本 77% 的邏輯
    metric_ours = FaceOcclusion(FACE_PKL)      # 排除 Label 2 的邏輯

    layout_list = safe_load(LAYOUT_PKL)
    print(f"成功載入 {len(layout_list)} 筆資料。\n")

    for i, data in enumerate(layout_list):
        bboxes = torch.tensor(data[0])
        labels = torch.tensor(data[1])

        # 【模式一：原本邏輯】只排除極大的畫布背景，其餘全算 (包含圖片 Label 2)
        mask_all = (bboxes[:, 2] < 0.99) | (bboxes[:, 3] < 0.99)
        metric_original.compute(bboxes[mask_all], i)

        # 【模式二：精準模式】排除 Label 2 (Image) 與 Label 3 (Canvas)
        # 只看文字 (Label 1) 與 SVG 元素 (Label 0) 是否蓋住臉
        mask_filtered = (labels != 2) & (labels != 3)
        filtered_bboxes = bboxes[mask_filtered]
        
        # 即使 filtered_bboxes 為空，也要傳入（metric 會處理），確保 index 對齊
        metric_ours.compute(filtered_bboxes, i)

    res_orig = metric_original.report()
    res_ours = metric_ours.report()

    print("="*55)
    print("【數據 A：全物件遮擋 (包含 Label 2 圖片)】")
    print(f"平均遮擋率: {res_orig['Mean_Occlusion_Rate']:.2%}")
    print(f"這就是你跑出的 77%，因為人臉本就在圖片裡。")
    
    print("-" * 55)
    
    print("【數據 B：精準遮擋 (排除 Label 2 圖片、Label 3 colorbackground) 論文使用】")
    print(f"平均文字/元素遮擋率 (Mean): {res_ours['Mean_Occlusion_Rate']:.2%}")
    print(f"嚴重遮擋頻率 (Freq > 10%): {res_ours['Occlusion_Frequency_10']:.2%}")
    print("註: 已成功排除圖片物件對其內部人臉的「誤判」遮擋。")
    print("="*55)

if __name__ == "__main__":
    run_evaluation()