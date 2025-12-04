from pathlib import Path
import pickle
import numpy as np

PKL_ROOT = Path("data/dataset/crello")

def check_split(split: str):
    pkl_path = PKL_ROOT / f"crello_{split}_imgmap.pkl"
    print(f"\n=== {split} ===")
    print("讀取", pkl_path)

    with pkl_path.open("rb") as f:
        layouts = pickle.load(f)

    all_boxes = []
    for boxes, labels in layouts:
        all_boxes.append(boxes)

    all_boxes = np.concatenate(all_boxes, axis=0)  # (M, 4)
    x, y, w, h = all_boxes.T

    print("x 範圍:", x.min(), "→", x.max())
    print("y 範圍:", y.min(), "→", y.max())
    print("w 範圍:", w.min(), "→", w.max())
    print("h 範圍:", h.min(), "→", h.max())

if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        check_split(split)
