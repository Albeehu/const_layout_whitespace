import os
import json
import csv
import argparse
import importlib.util

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def load_module(script_path: str):
    spec = importlib.util.spec_from_file_location("t53", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify_scores(L, R, T, Bm):
    margins = torch.tensor([L, R, T, Bm], dtype=torch.float32)
    S_frame = (margins.mean() - 0.5 * (margins.std(unbiased=False) + 1e-6)).clamp(0.0, 1.0)

    h = torch.tensor([L, R], dtype=torch.float32)
    v = torch.tensor([T, Bm], dtype=torch.float32)

    h_max = h.max()
    v_max = v.max()

    S_side = (h_max + 0.8 * (h_max - h.min()) + 0.2 * v.mean()).clamp(0, 1)
    S_tb = (v_max + 0.8 * (v_max - v.min()) + 0.2 * h.mean()).clamp(0, 1)
    S_hybrid = torch.sqrt(S_side * S_tb).clamp(0, 1)

    scores = torch.stack([S_frame, S_side, S_tb, S_hybrid])
    idx = int(scores.argmax().item())
    names = ["frame", "side", "top-bottom", "hybrid"]
    return names[idx], [float(x.item()) for x in scores]


def compute_metrics_for_one(t53, bbox_one, label_one, valid_mask_one):
    valid = valid_mask_one & (label_one != t53.FACE_ID) & (label_one != t53.BG_ID) & (label_one != t53.MASK_ID)
    if int(valid.sum().item()) == 0:
        return None

    xyxy = t53.xywh_to_xyxy(bbox_one[valid])
    x1, y1, x2, y2 = xyxy.unbind(-1)

    areas = ((x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0))
    wr = float((1.0 - areas.sum().clamp(0.0, 1.0)).item())

    L = float(x1.min().clamp(0.001, 0.999).item())
    R = float((1.0 - x2.max()).clamp(0.001, 0.999).item())
    T = float(y1.min().clamp(0.001, 0.999).item())
    Bm = float((1.0 - y2.max()).clamp(0.001, 0.999).item())

    margins = {"L": L, "R": R, "T": T, "B": Bm}
    max_margin_name = max(margins, key=margins.get)

    style_name, scores = classify_scores(L, R, T, Bm)

    return {
        "wr": wr,
        "L": L,
        "R": R,
        "T": T,
        "B": Bm,
        "max_margin": max_margin_name,
        "pred_style": style_name,
        "S_frame": scores[0],
        "S_side": scores[1],
        "S_top_bottom": scores[2],
        "S_hybrid": scores[3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, default="train_fixed_v6_0.6pkl_v53.py")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--dataset", type=str, default="crello")
    ap.add_argument("--single_pkl", type=int, default=0)
    ap.add_argument("--mix_weights", type=float, nargs=3, default=[0.15, 0.35, 0.50])
    ap.add_argument("--mix_epoch_size", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_samples", type=int, default=400)
    ap.add_argument("--max_nodes", type=int, default=5)
    ap.add_argument("--max_faces", type=int, default=4)
    ap.add_argument("--fix_aspect", type=int, default=1)
    ap.add_argument("--fix_text_aspect", type=int, default=0)
    ap.add_argument("--freeze_face", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t53 = load_module(args.script)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dataset = t53.get_dataset(args.dataset, 'train')

    def _build_raw_dataset_from_pkl(pkl_path: str):
        import pickle
        with open(pkl_path, 'rb') as f:
            data_list = pickle.load(f)
        return t53.RawLayoutDataset(
            data_list,
            num_classes=base_dataset.num_classes,
            max_nodes=args.max_nodes,
            max_faces=args.max_faces,
            colors=base_dataset.colors,
            aug_hflip=False,
            aug_vflip=False,
            flip_prob=0.0,
        )

    if args.single_pkl == 1:
        ds = _build_raw_dataset_from_pkl(t53.PKL_PATH_WS_60)
    else:
        mixed = [
            _build_raw_dataset_from_pkl(t53.PKL_PATH_CRELLO_FULL),
            _build_raw_dataset_from_pkl(t53.PKL_PATH_HIGH_QUALITY),
            _build_raw_dataset_from_pkl(t53.PKL_PATH_WS_60),
        ]
        ds = t53.WeightedMixedRawDataset(
            mixed,
            source_weights=args.mix_weights,
            epoch_size=args.num_samples,
            seed=args.seed,
        )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    netG = t53.Generator(4, base_dataset.num_classes, d_model=256, nhead=4, num_layers=4).to(device)
    netD = t53.Discriminator(base_dataset.num_classes, d_model=256, nhead=4, num_layers=4).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=1e-4)
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4)

    t53.load_training_checkpoint(args.ckpt, device, netG, netD, optG, optD)
    netG.eval()

    rows = []
    style_counts = {"frame": 0, "side": 0, "top-bottom": 0, "hybrid": 0}
    max_margin_counts = {"L": 0, "R": 0, "T": 0, "B": 0}
    total = 0

    with torch.no_grad():
        for batch in dl:
            label = batch["x"].to(device)
            pos = batch["pos"].to(device)
            mask = batch["mask"].to(device)
            padding = ~mask

            B = label.size(0)
            z = torch.rand((B, label.size(1), 4), device=device)
            bbox = torch.clamp(netG(z, label, padding), 0, 1)

            if args.fix_aspect:
                aspect_target_ids = (t53.SVG_ID, t53.IMG_ID) if not args.fix_text_aspect else (t53.SVG_ID, t53.TEXT_ID, t53.IMG_ID)
                bbox = t53.project_fixed_aspect_scale(
                    bbox, pos, label, padding, target_ids=aspect_target_ids
                )

            if args.freeze_face:
                face2img = t53.infer_face2img_from_reference(
                    pos, label, padding, max_faces=args.max_faces, contain_thr=0.98
                )
                face_mask = batch["face_mask"].to(device)
                face_rel = batch["face_rel"].to(device)
                bbox = t53.hard_couple_faces_to_images(
                    bbox, label, padding,
                    face2img=face2img,
                    face_mask=face_mask,
                    face_rel_gt=face_rel,
                    use_gt_face_rel=True,
                    keep_inside_image=True,
                )

            valid_mask = mask.bool()

            for i in range(B):
                if total >= args.num_samples:
                    break
                m = compute_metrics_for_one(
                    t53,
                    bbox[i].cpu(),
                    label[i].cpu(),
                    valid_mask[i].cpu(),
                )
                if m is None:
                    continue
                m["sample_id"] = total
                rows.append(m)
                style_counts[m["pred_style"]] += 1
                max_margin_counts[m["max_margin"]] += 1
                total += 1

            if total >= args.num_samples:
                break

    if len(rows) == 0:
        raise RuntimeError("No valid samples were collected.")

    csv_path = os.path.join(args.out_dir, "sample_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    wrs = [r["wr"] for r in rows]
    summary = {
        "num_samples": total,
        "wr_mean": float(np.mean(wrs)),
        "wr_std": float(np.std(wrs)),
        "wr_ge_0.60_rate": float(np.mean(np.array(wrs) >= 0.60)),
        "style_counts": style_counts,
        "style_ratios": {k: v / max(total, 1) for k, v in style_counts.items()},
        "max_margin_counts": max_margin_counts,
        "max_margin_ratios": {k: v / max(total, 1) for k, v in max_margin_counts.items()},
        "margin_mean": {
            "L": float(np.mean([r["L"] for r in rows])),
            "R": float(np.mean([r["R"] for r in rows])),
            "T": float(np.mean([r["T"] for r in rows])),
            "B": float(np.mean([r["B"] for r in rows])),
        },
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"saved: {csv_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
