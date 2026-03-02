#用來檢查任何pkl (list[boxes, labels])
import pickle, numpy as np, sys
FACE_ID = 5

pkl_path = sys.argv[1]
data = pickle.load(open(pkl_path, "rb"))

num_samples = len(data)
with_face = 0
face_tokens = 0
face_per_sample = []

for boxes, labels in data:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    c = int((labels == FACE_ID).sum())
    face_tokens += c
    face_per_sample.append(c)
    if c > 0:
        with_face += 1

print("pkl:", pkl_path)
print("samples:", num_samples)
print("samples_with_face:", with_face, f"({with_face/num_samples*100:.2f}%)")
print("total_face_tokens:", face_tokens)
print("min/mean/max face per sample:", min(face_per_sample), sum(face_per_sample)/num_samples, max(face_per_sample))

if with_face == num_samples:
    print("[OK] 每筆都有 face")
else:
    print("[WARN] 不是每筆都有 face（若你要做遮擋證明，建議只保留有 face 的樣本）")