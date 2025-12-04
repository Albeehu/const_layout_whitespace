#用來畫pkl中的layout
import pickle
from pathlib import Path
from data.crello import CrelloDataset
from util import save_image  # 用你 train.py 裡用的那個 save_image

# 讀生成的 layout
with open("/home/albee/const_layout_test/data/dataset/crello/crello_test_face_boxes.pkl", "rb") as f:
    layouts = pickle.load(f)

print("共有", len(layouts), "個 layout")

# 要顏色跟 label 名稱，可以借用 CrelloDataset
ds = CrelloDataset('train')
colors = ds.colors

out_dir = Path("output/crello/test_face")
out_dir.mkdir(parents=True, exist_ok=True)

# 例如存前 16 張，每張一個 png
for idx, (boxes, labels) in enumerate(layouts[:100]):
    # 這裡簡單包成一個假的 batch，方便重用 save_image
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import to_dense_batch

    x = torch.tensor(boxes, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, y=y)
    data.batch = torch.zeros(y.size(0), dtype=torch.long)  # 全部同一張圖

    bbox_real, mask = to_dense_batch(data.x, data.batch)
    label, _ = to_dense_batch(data.y, data.batch)

    out_path = out_dir / f"gen_{idx:04d}.png"
    save_image(bbox_real, label, mask, colors, out_path)
    print("saved", out_path)
