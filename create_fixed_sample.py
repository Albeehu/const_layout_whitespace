#產新的fixed_sample_v18.pt
import torch
import numpy as np

def generate_v18_fixed_sample():
    out_path = 'fixed_sample_v18.pt'
    batch_size = 64
    max_nodes = 6
    max_faces = 4
    latent_size = 4
    total_cap = max_nodes + max_faces # 總長度 10

    # 1. 初始化張量 (標籤預設為 4-Mask)
    fixed_label = torch.full((batch_size, total_cap), 4, dtype=torch.long)
    fixed_mask = torch.zeros((batch_size, total_cap), dtype=torch.bool)
    fixed_z = torch.randn((batch_size, total_cap, latent_size))

    for i in range(batch_size):
        # 生成 3-6 個一般元件
        num_o = np.random.randint(3, max_nodes + 1)
        o_labels = np.random.choice([0, 1, 2, 3], size=num_o).tolist()
        if 2 not in o_labels: o_labels[0] = 2 # 確保有 Image 容器
        
        # 生成 1-4 個人臉
        num_f = np.random.randint(1, max_faces + 1)
        
        # 按照 v18 Dataset 的拼接邏輯：[設計元件..., 人臉...]
        # 填充一般元件
        fixed_label[i, :num_o] = torch.LongTensor(o_labels)
        fixed_mask[i, :num_o] = True
        
        # 人臉接在後面
        fixed_label[i, num_o : num_o + num_f] = 5
        fixed_mask[i, num_o : num_o + num_f] = True

    # 核心：使用 torch.save 產生真正的二進位檔案
    save_data = {'label': fixed_label, 'mask': fixed_mask, 'z': fixed_z}
    torch.save(save_data, out_path)
    print(f" 成功產出二進位檔案: {out_path}")

if __name__ == "__main__":
    generate_v18_fixed_sample()