#debug 確認權重層數
import torch

# 載入權重檔（注意 PyTorch 2.6 以上需加 weights_only=False）
checkpoint = torch.load('/home/albee/const_layout_whitespace/pretrained/layoutnet_crello.pth.tar', map_location='cpu', weights_only=False)

# 如果你的 checkpoint 是字典格式，通常權重在 'state_dict' 鍵值下
# 如果直接是權重，就直接使用 checkpoint
if 'state_dict' in checkpoint:
    sd = checkpoint['state_dict']
else:
    sd = checkpoint

# 過濾出所有 Transformer 層的索引
# 假設命名規則包含 'layers.數字'
layers = set()
for key in sd.keys():
    if 'transformer' in key and 'layers.' in key:
        # 提取 layers. 後面的數字
        parts = key.split('layers.')
        if len(parts) > 1:
            layer_num = parts[1].split('.')[0]
            layers.add(int(layer_num))

print(f"找到的 Transformer 總層數: {len(layers)}")
print(f"層數索引範圍: {min(layers) if layers else 'N/A'} 到 {max(layers) if layers else 'N/A'}")

# 順便確認 d_model (隱藏層維度)
if 'emb_label.weight' in sd:
    print(f"Label Embedding 維度 (d_model): {sd['emb_label.weight'].shape[1]}")