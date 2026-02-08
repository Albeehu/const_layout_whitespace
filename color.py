#用來判斷匡對應的顏色
import torch
from data import get_dataset  # 確保你的環境可以 import 到專案內的 data 模組

def check_label_colors():
    # 1. 載入資料集 (以 crello 為例)
    try:
        # 這裡只需要載入資料集物件，不需要真正跑 dataloader
        dataset = get_dataset('crello', 'train')
        
        # 2. 取得類別名稱與顏色
        # LayoutGAN++ 的資料集通常會把顏色存放在 dataset.colors 
        # 類別名稱通常在 dataset.categories 或 dataset.num_classes 相關定義中
        categories = ["Svgelement", "Textelement", "Imageelement", 
                      "colorbackground", "Svgmaskelement", "face"]
        
        colors = dataset.colors if hasattr(dataset, 'colors') else None
        
        print("-" * 50)
        print(f"{'ID':<5} | {'Category Name':<18} | {'RGB Color (0-255)':<15}")
        print("-" * 50)
        
        if colors is not None:
            for i, name in enumerate(categories):
                # dataset.colors 通常是 [0, 1] 的 float，轉換成 0-255 比較好辨認
                c = colors[i]
                rgb = (int(c[0]*255), int(c[1]*255), int(c[2]*255))
                print(f"{i:<5} | {name:<18} | {rgb}")
        else:
            print("找不到 dataset.colors 屬性，請檢查 data/crello.py 的定義。")
            
        print("-" * 50)

    except Exception as e:
        print(f"錯誤: {e}")
        print("請確保你在專案根目錄執行此腳本，且環境中已安裝必要的 dependency。")

if __name__ == "__main__":
    check_label_colors()