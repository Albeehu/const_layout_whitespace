#用來看crello的label對應的顏色
from datasets import load_dataset

# 下載 Crello dataset（train split 就夠看 label 了）
ds = load_dataset("cyberagent/crello", split="train", revision="5.0.0")

# "type" 這個欄位是 Sequence(ClassLabel)，feature 裡有 names
type_feature = ds.features["type"].feature

print(type_feature)          # 看一下型別（ClassLabel）
print("num classes:", type_feature.num_classes)
print("names:", type_feature.names)

print("\nID -> name 對照：")
for i, name in enumerate(type_feature.names):
    print(i, "->", name)
