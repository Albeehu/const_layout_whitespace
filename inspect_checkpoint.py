#看checkpoint的內容
import torch
from pathlib import Path
from pprint import pprint

# TODO: 把這裡改成你想查的 checkpoint 路徑
ckpt_path = Path("/home/albee/const_layout_whitespace/output/crello_mainpart_face/LayoutGAN++/ws_loss_test/checkpoint.pth.tar")

ckpt = torch.load(ckpt_path, map_location="cpu")

print("=== checkpoint keys ===")
print(list(ckpt.keys()))

# 常見情況 1：裡面有 'args'
if "args" in ckpt:
    print("\n=== args (training command options) ===")
    pprint(ckpt["args"])

# 常見情況 2：有人用 'config' 或 'hparams'
elif "config" in ckpt:
    print("\n=== config ===")
    pprint(ckpt["config"])
elif "hparams" in ckpt:
    print("\n=== hparams ===")
    pprint(ckpt["hparams"])
else:
    print("\n[!] 找不到 args / config / hparams 相關資訊，可能當初沒存。")
