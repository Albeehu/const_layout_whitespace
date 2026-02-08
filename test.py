"""
串到metric.py跑FID、face occlusion rate的實驗
用來比較原始模型跟我的模型的差別 可放論文的experiment
"""
import torch
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
from metric import LayoutFID, FaceOcclusion # 確保 metric.py 在同一個目錄

# ================= 1. 設定路徑 =================
MODEL_WEIGHTS = "model_weights/best_model.pth" # 訓練完後的權重路徑
TEST_DATA_PKL = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_test.pkl"# 原始測試集佈局
FACE_DATA_PKL = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_test_face.pkl"# 你上傳的臉部標註
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluation(mode="baseline"):
    print(f"--- 正在啟動 {mode.upper()} 評估模式 ---")
    
    # 2. 初始化指標類別
    fid_metric = LayoutFID(dataset_name='crello', device=DEVICE)
    face_metric = FaceOcclusion(FACE_DATA_PKL, device=DEVICE)
    
    # 3. 載入原始測試集資料
    with open(TEST_DATA_PKL, 'rb') as f:
        test_data_list = pickle.load(f)
    
    # 如果你的模型需要 DataLoader，可以在這裡轉換
    # test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)

    # 4. 開始迴圈
    for i, data in enumerate(test_data_list):
        # data 通常包含 'bboxes' (N, 4), 'labels' (N,)
        
        if mode == "baseline":
            # 直接使用原始資料的座標作為預測值
            pred_bboxes = torch.tensor(data['bboxes']).to(DEVICE)
            labels = torch.tensor(data['labels']).to(DEVICE)
        else:
            # --- 模式 B: 模型推論 ---
            # model = YourModel().to(DEVICE)
            # model.load_state_dict(torch.load(MODEL_WEIGHTS))
            # model.eval()
            # with torch.no_grad():
            #     pred_bboxes = model(data) # 假設模型輸出 [N, 4]
            # labels = data['labels']
            pass

        # 5. 更新指標
        # 更新 FID 特徵 (這需要 padding_mask，這裡簡化處理)
        mask = torch.ones(pred_bboxes.shape[0], dtype=torch.bool).to(DEVICE)
        fid_metric.collect_features(pred_bboxes.unsqueeze(0), labels.unsqueeze(0), mask.unsqueeze(0), real=(mode=="baseline"))
        
        # 更新臉部遮擋率
        face_metric.compute(pred_bboxes, i)

    # 6. 印出報告
    print("\n" + "="*30)
    print(f"測試樣本總數: {len(test_data_list)}")
    
    if mode == "baseline":
        # 存下真實特徵供未來 FID 比較
        # fid_metric.save_real_features("real_feats.pkl")
        print("已完成原始資料特徵收集")
    else:
        print(f"Layout FID: {fid_metric.compute_fid():.4f}")
        
    results = face_metric.report()
    print(f"平均臉部遮擋率 (Mean Occlusion): {results['Mean_Occlusion_Rate']:.2%}")
    print(f"嚴重遮擋頻率 (>10%): {results['Occlusion_Frequency_10']:.2%}")
    print("="*30)

if __name__ == "__main__":
    # 你現在可以先跑這個來拿 Baseline 數據！
    run_evaluation(mode="baseline")