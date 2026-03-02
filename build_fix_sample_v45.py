import torch
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "fixed_sample.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fixed_sample_v18.pt"
K = 4  # max_nodes

ck = torch.load(SRC, map_location="cpu", weights_only=False)

label = ck["label"]
mask  = ck["mask"]

# dtype normalize
label = torch.round(label).long() if label.dtype.is_floating_point else label.long()
mask  = mask.bool()

B, N = label.shape

# latent z
if "z" in ck and ck["z"].dim() == 3:
    z_old = ck["z"].float()              # (B,N,L)
    L = z_old.size(-1)
else:
    z_old = None
    L = 256  # 如果原檔沒有 z，就用你訓練的 latent_size

label_new = torch.zeros(B, K, dtype=torch.long)
mask_new  = torch.zeros(B, K, dtype=torch.bool)
z_new     = torch.randn(B, K, L, dtype=torch.float32)

for b in range(B):
    valid = torch.nonzero(mask[b], as_tuple=False).flatten()
    take = valid[:K]
    m = take.numel()
    if m == 0:
        continue
    label_new[b, :m] = label[b, take]
    mask_new[b, :m] = True
    if z_old is not None:
        z_new[b, :m] = z_old[b, take]

torch.save({"label": label_new, "mask": mask_new, "z": z_new}, OUT)
print("saved:", OUT)
print(" label:", tuple(label_new.shape), label_new.dtype)
print(" mask :", tuple(mask_new.shape), mask_new.dtype)
print(" z    :", tuple(z_new.shape), z_new.dtype)