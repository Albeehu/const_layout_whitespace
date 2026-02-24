import torch

def inspect_fixed_sample(file_path):
    print(f"--- 正在檢查: {file_path} ---")
    try:
        # 載入固定樣本
        data = torch.load(file_path, map_location='cpu')
        
        # 檢查主要的 keys
        labels = data.get('label')  # [B, N]
        masks = data.get('mask')    # [B, N]
        z = data.get('z')           # [B, N, 4]
        
        if labels is None or masks is None:
            print("錯誤：檔案中找不到 label 或 mask。")
            return

        batch_size = labels.size(0)
        num_elements = labels.size(1)
        
        print(f"Batch Size: {batch_size}")
        print(f"Max Elements per Sample: {num_elements}")
        print("-" * 30)

        # 遍歷每個 Batch 檢查標籤組合
        for i in range(batch_size):
            # 只取出有效元件的標籤 (mask 為 True 的部分)
            valid_mask = masks[i]
            sample_labels = labels[i][valid_mask].tolist()
            
            has_face = 5 in sample_labels
            has_image = 2 in sample_labels
            
            print(f"Sample {i:02d}:")
            print(f"  有效元件數量: {len(sample_labels)}")
            print(f"  標籤清單: {sample_labels}")
            
            # 診斷：是否有 Face 但沒 Image
            if has_face and not has_image:
                print("  >> [警報] 此樣本包含 Face (5) 但缺少 Image (2)！")
            elif has_face and has_image:
                print("  >> [正常] Face 與 Image 同時存在。")
            
    except Exception as e:
        print(f"讀取失敗: {e}")

if __name__ == "__main__":
    # 請確保路徑正確
    target_file = '/home/albee/const_layout_whitespace/fixed_sample.pt' 
    inspect_fixed_sample(target_file)