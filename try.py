import pickle

path = "/home/albee/const_layout_test/data/dataset/crello/crello_train_all.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print("總 layout 筆數 =", len(data))
print("type(data) =", type(data))

first = data[0]
print("type(first) =", type(first))

# 如果是 tuple，看看裡面有幾個欄位
if isinstance(first, tuple):
    print("第一筆是 tuple，長度 =", len(first))
    for i, part in enumerate(first):
        print(f"\n--- 第 {i} 個元素 ---")
        print("type:", type(part))
        # 盡量印一點內容出來看
        try:
            print("len(part) =", len(part))
        except TypeError:
            pass
        print("repr(part) =", repr(part)[:300])
else:
    print("第一筆內容 =", first)
