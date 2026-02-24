import torch
from model.layoutganpp import Generator

# 1. 讀取預訓練檔
checkpoint = torch.load('pretrained/layoutnet_crello.pth.tar', map_location='cpu', weights_only=False)
old_keys = list(checkpoint['state_dict'].keys()) if 'state_dict' in checkpoint else list(checkpoint.keys())

# 2. 根據剛剛抓到的參數名稱實例化 (對齊 v18 邏輯)
# dim_latent=4 (args.latent_size), num_label=6 (0-5)
netG = Generator(dim_latent=4, num_label=6, d_model=256, nhead=4, num_layers=4)
new_keys = list(netG.state_dict().keys())

print("\n--- 預訓練權重前 10 個 Keys ---")
for k in old_keys[:10]: print(k)

print("\n--- 你目前模型前 10 個 Keys ---")
for k in new_keys[:10]: print(k)