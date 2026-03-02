import pickle
path = "/home/albee/const_layout_whitespace/data/dataset/crello/crello_train_all.pkl"
with open(path, 'rb') as f:
    data = pickle.load(f)

# 看看第一個樣本有沒有 ID 或其他路徑資訊
sample = data[0]
if isinstance(sample, dict):
    print("欄位名稱：", sample.keys())
else:
    # 如果是元組，看看除了 BBox 和 Label 外還有什麼
    print(f"樣本共有 {len(sample)} 個物件")
    for i in range(len(sample)):
        print(f"Index {i} 的類型: {type(sample[i])}")