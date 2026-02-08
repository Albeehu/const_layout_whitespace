#確認原始的label是否有重疊 判斷fake是否學錯
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch_geometric.loader import DataLoader
from torchvision import transforms as T
from data import get_dataset  # 確保你的 data.py 在同一個目錄下

def visual_check_labels(dataset_name='crello', num_samples=10, output_dir='debug_labels'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 載入原始資料集
    print(f"正在載入 {dataset_name} 驗證集...")
    dataset = get_dataset(dataset_name, 'val') # 使用 val 比較快且通常比較乾淨
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 取得顏色定義 (如果有)
    colors = getattr(dataset, 'colors', None)

    print(f"開始生成 {num_samples} 張檢查圖...")
    
    for i, data in enumerate(dataloader):
        if i >= num_samples:
            break
        
        # 2. 提取座標與標籤
        # 根據你的 train code，座標是 [cx, cy, w, h]
        boxes = data.x.numpy()  # [N, 4]
        labels = data.y.numpy() # [N]
        
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0) # 佈局通常 Y 軸向下
        ax.set_aspect('equal')

        for j in range(len(boxes)):
            cx, cy, w, h = boxes[j]
            if w <= 0 or h <= 0: continue # 跳過 padding
            
            # 轉換為左上角座標用於繪圖
            x1, y1 = cx - w/2, cy - h/2
            
            # 取得顏色
            edge_color = 'blue'
            if colors and labels[j] < len(colors):
                # 將 0-255 轉為 0-1
                edge_color = [c/255.0 for c in colors[labels[j]]]
            
            # 繪製方框
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                     edgecolor=edge_color, facecolor=edge_color, alpha=0.2)
            ax.add_patch(rect)
            
            # 標註類別 ID
            ax.text(x1, y1, str(labels[j]), fontsize=8, color='black', fontweight='bold')

        plt.title(f"Real Label Sample {i}")
        save_path = os.path.join(output_dir, f"real_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"已儲存: {save_path}")

if __name__ == "__main__":
    # 你可以修改 dataset 名稱
    visual_check_labels(dataset_name='crello', num_samples=10)