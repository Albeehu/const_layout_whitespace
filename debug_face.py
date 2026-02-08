#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
debug_face.py (single-checkpoint sanity generator + ID overlay)

✅ Works with your setup:
- CrelloDataset returns torch_geometric.data.Data
- Generator (model/layoutganpp.py) uses nn.Embedding(label) => label MUST be Long indices
- Saves PNGs + metrics.json
- Can overlay token id (elem_id) to inspect overlaps
- Default skips drawing/background-like classes 3,4 (ColoredBackground, SvgMaskElement)

Class ids (as you provided):
0 SvgElement
1 TextElement
2 ImageElement
3 ColoredBackground
4 SvgMaskElement
5 face
"""

import os
import sys
import json
import argparse
import random
import importlib
import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# torch_geometric (required)
try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.data import Batch as GeoBatch
    from torch_geometric.utils import to_dense_batch
    HAS_PYG = True
except Exception:
    GeoData, GeoBatch, to_dense_batch = None, None, None
    HAS_PYG = False


# -------------------------
# utils
# -------------------------

CLASS_NAMES = {
    0: "SvgElement",
    1: "TextElement",
    2: "ImageElement",
    3: "ColoredBackground",
    4: "SvgMaskElement",
    5: "face",
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)

def parse_int_set(s: str) -> Set[int]:
    s = (s or "").strip()
    if not s:
        return set()
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.add(int(part))
    return out

def strip_prefix_if_present(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """
    Supports common formats:
      - {'netG': state_dict}
      - {'netG_state_dict': ...}
      - {'state_dict': ...} (possibly with prefixes)
      - state_dict directly
    """
    if isinstance(ckpt, dict):
        for k in ["netG", "G", "generator", "netG_state_dict", "G_state_dict", "state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # might already be a state dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise ValueError("Checkpoint format not recognized.")

def try_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


# -------------------------
# bbox / IoU
# -------------------------

def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(dim=-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)

def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(dim=-1)
    x0 = x
    y0 = y
    x1 = x + w
    y1 = y + h
    return torch.stack([x0, y0, x1, y1], dim=-1)

def bbox_to_xyxy(box: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "cxcywh":
        return cxcywh_to_xyxy(box)
    if fmt == "xywh":
        return xywh_to_xyxy(box)
    if fmt == "xyxy":
        return box
    raise ValueError(f"Unknown bbox_format: {fmt}")

def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [...,4], b: [...,4] broadcastable, xyxy
    returns IoU [...]
    """
    ax0, ay0, ax1, ay1 = a.unbind(-1)
    bx0, by0, bx1, by1 = b.unbind(-1)

    inter_x0 = torch.maximum(ax0, bx0)
    inter_y0 = torch.maximum(ay0, by0)
    inter_x1 = torch.minimum(ax1, bx1)
    inter_y1 = torch.minimum(ay1, by1)

    inter_w = torch.clamp(inter_x1 - inter_x0, min=0.0)
    inter_h = torch.clamp(inter_y1 - inter_y0, min=0.0)
    inter = inter_w * inter_h

    area_a = torch.clamp(ax1 - ax0, min=0.0) * torch.clamp(ay1 - ay0, min=0.0)
    area_b = torch.clamp(bx1 - bx0, min=0.0) * torch.clamp(by1 - by0, min=0.0)
    union = area_a + area_b - inter + 1e-8
    return inter / union


# -------------------------
# dataset loading (robust)
# -------------------------

def scan_for_crello_modules(repo_root: Path) -> List[str]:
    """
    Find python files containing 'class CrelloDataset' and convert to dotted module path.
    This avoids spec_from_file_location problems with relative imports.
    """
    modules = []
    for fp in repo_root.rglob("*.py"):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "class CrelloDataset" not in txt:
            continue

        rel = fp.relative_to(repo_root).with_suffix("")  # remove .py
        mod = ".".join(rel.parts)  # data/dataset/crello -> data.dataset.crello
        modules.append(mod)
    return modules

def build_dataset(args) -> Any:
    """
    Tries:
      1) explicit --dataset_module/--dataset_class if provided
      2) common candidates
      3) scan repo for a module containing CrelloDataset and import it as dotted module
    """
    dataset_name = args.dataset
    split = args.split
    use_face = (dataset_name == "crello_mainpart_face")
    variant = args.variant

    # 1) explicit module/class
    if args.dataset_module:
        m = importlib.import_module(args.dataset_module)
        cls_name = args.dataset_class or "CrelloDataset"
        if not hasattr(m, cls_name):
            raise ImportError(f"--dataset_module {args.dataset_module} has no class {cls_name}")
        CrelloDataset = getattr(m, cls_name)
        sig = inspect.signature(CrelloDataset.__init__)
        kwargs = {}
        if "split" in sig.parameters: kwargs["split"] = split
        if "variant" in sig.parameters: kwargs["variant"] = variant
        if "use_face" in sig.parameters: kwargs["use_face"] = use_face
        return CrelloDataset(**kwargs)

    # 2) common candidates
    candidates = [
        "data.dataset.crello",
        "data.dataset.crello_dataset",
        "data.dataset.crello_pyg",
        "data.datasets.crello",
        "data.crello",
        "dataset.crello",
        "datasets.crello",
    ]
    last_err = None
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
            if not hasattr(m, "CrelloDataset"):
                continue
            CrelloDataset = getattr(m, "CrelloDataset")
            sig = inspect.signature(CrelloDataset.__init__)
            kwargs = {}
            if "split" in sig.parameters: kwargs["split"] = split
            if "variant" in sig.parameters: kwargs["variant"] = variant
            if "use_face" in sig.parameters: kwargs["use_face"] = use_face
            return CrelloDataset(**kwargs)
        except Exception as e:
            last_err = e

    # 3) scan
    repo_root = Path(__file__).resolve().parent
    found = scan_for_crello_modules(repo_root)
    # prefer shorter paths first
    found = sorted(found, key=lambda s: (s.count("."), len(s)))
    for mod in found:
        try:
            m = importlib.import_module(mod)
            if not hasattr(m, "CrelloDataset"):
                continue
            CrelloDataset = getattr(m, "CrelloDataset")
            sig = inspect.signature(CrelloDataset.__init__)
            kwargs = {}
            if "split" in sig.parameters: kwargs["split"] = split
            if "variant" in sig.parameters: kwargs["variant"] = variant
            if "use_face" in sig.parameters: kwargs["use_face"] = use_face
            print(f"[build_dataset] Using CrelloDataset from: {mod}")
            return CrelloDataset(**kwargs)
        except Exception as e:
            last_err = e
            continue

    raise ImportError(
        "Cannot import/build CrelloDataset. "
        "Tried explicit module, common candidates, and scanned repo for 'class CrelloDataset'. "
        f"Last error: {repr(last_err)}"
    )


# -------------------------
# Generator build
# -------------------------

def infer_num_classes_from_ckpt(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    for k, v in sd.items():
        if torch.is_tensor(v) and k.endswith("emb_label.weight") and v.dim() == 2:
            return int(v.size(0))
    return None

def infer_d_model_from_ckpt(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    # self_attn.in_proj_weight: [3*d_model, d_model]
    for k, v in sd.items():
        if torch.is_tensor(v) and k.endswith("self_attn.in_proj_weight") and v.dim() == 2:
            return int(v.size(1))
    return None

def infer_max_seq_from_ckpt(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        kl = k.lower()
        if ("pos" in kl and ("emb" in kl or "embed" in kl)) or ("position" in kl and ("emb" in kl or "embed" in kl)):
            if v.dim() == 2:
                return int(v.size(0))
            if v.dim() == 3:
                return int(v.size(1))
    return None

def build_generator(args, num_classes: int, sd: Dict[str, torch.Tensor]) -> torch.nn.Module:
    import torch.nn as nn

    m = importlib.import_module(args.G_module)

    # pick class
    if args.G_class:
        if not hasattr(m, args.G_class):
            raise ImportError(f"{args.G_module} has no class {args.G_class}")
        G = getattr(m, args.G_class)
    else:
        G = None
        for cname in ["Generator", "NetG", "LayoutGANpp", "LayoutGANPP", "LayoutGANPPGenerator", "LayoutGANppGenerator"]:
            if hasattr(m, cname):
                G = getattr(m, cname)
                break
        if G is None:
            for name, obj in vars(m).items():
                if isinstance(obj, type) and issubclass(obj, nn.Module):
                    G = obj
                    break
        if G is None:
            raise ImportError(f"No nn.Module class found in {args.G_module}")

    sig = inspect.signature(G.__init__)
    inferred_d_model = infer_d_model_from_ckpt(sd)
    inferred_max_seq = infer_max_seq_from_ckpt(sd)

    d_model = args.G_d_model
    if inferred_d_model is not None and inferred_d_model != d_model:
        print(f"[warn] ckpt d_model={inferred_d_model} != --G_d_model {d_model}. "
              f"If you see shape mismatch, set --G_d_model {inferred_d_model}.")

    kwargs = {}

    def set_first(cands: List[str], val):
        for pn in cands:
            if pn in sig.parameters and pn not in kwargs:
                kwargs[pn] = val
                return pn
        return None

    # core
    set_first(["num_classes", "num_class", "n_class", "n_classes", "num_label", "n_labels"], num_classes)
    set_first(["dim_latent", "latent_size", "latent_dim", "z_dim", "dim_z", "noise_dim"], args.latent_size)

    DMODEL_CANDS = [
        "d_model", "dim_model", "model_dim",
        "embed_dim", "emb_dim", "dim_embed", "d_embed",
        "hidden_size", "dim_hidden", "d_hidden", "hid_dim",
        "transformer_dim", "dim_transformer",
    ]
    set_first(DMODEL_CANDS, d_model)
    set_first(["nhead", "num_heads", "n_head", "heads"], args.G_nhead)
    set_first(["num_layers", "n_layers", "nlayer", "n_layer", "depth"], args.G_num_layers)
    set_first(["bbox_dim", "dim_bbox", "box_dim"], 4)

    if inferred_max_seq is not None:
        set_first(
            ["max_len", "max_seq_len", "max_seq_length", "max_bbox", "max_bboxes", "max_elements", "max_num_elements"],
            inferred_max_seq
        )

    if any(p in sig.parameters for p in DMODEL_CANDS) and not any(p in kwargs for p in DMODEL_CANDS):
        raise TypeError(f"Generator expects model-dim (one of {DMODEL_CANDS}) but we didn't set it. Signature: {sig}")

    try:
        netG = G(**kwargs)
    except TypeError as e:
        raise TypeError(f"Failed to init Generator with kwargs={kwargs}. Signature: {sig}. Error: {repr(e)}")

    print(f"[build_generator] {args.G_module}.{netG.__class__.__name__} kwargs={kwargs}")
    return netG


# -------------------------
# batch parsing (PyG -> dense)
# -------------------------

@dataclass
class BatchPack:
    bbox_gt: Optional[torch.Tensor]       # [B,N,4]
    label: torch.Tensor                   # [B,N] or [B,N,C]
    padding_mask: torch.Tensor            # [B,N] bool, True=PAD
    face_bbox_gt: Optional[torch.Tensor]  # [B,4] optional

def parse_batch(raw: Any) -> BatchPack:
    if not (HAS_PYG and isinstance(raw, (GeoBatch, GeoData))):
        raise TypeError(f"Expected PyG Batch/Data but got {type(raw)}")

    b = raw

    def pick_attr(names: List[str]) -> Optional[torch.Tensor]:
        for n in names:
            if hasattr(b, n):
                v = getattr(b, n)
                if torch.is_tensor(v):
                    return v
        return None

    bbox_node = pick_attr(["bbox", "bboxes", "boxes", "x"])
    if bbox_node is None:
        raise ValueError("PyG batch has no bbox tensor. Tried: bbox/bboxes/boxes/x")
    if bbox_node.dim() != 2 or bbox_node.size(-1) != 4:
        raise ValueError(f"bbox_node must be [num_nodes,4], got {tuple(bbox_node.shape)}")

    label_node = pick_attr(["y", "label", "labels"])
    if label_node is None:
        raise ValueError("PyG batch has no label tensor. Tried: y/label/labels")

    batch_vec = b.batch if hasattr(b, "batch") and b.batch is not None else torch.zeros(
        (bbox_node.size(0),), dtype=torch.long, device=bbox_node.device
    )

    bbox_dense, valid = to_dense_batch(bbox_node, batch_vec)   # valid=True -> real token
    label_dense, _ = to_dense_batch(label_node, batch_vec)
    padding_mask = ~valid                                      # True=PAD

    face_bbox_gt = pick_attr(["face_bbox", "face_box", "face_bbox_gt", "gt_face_bbox"])
    if face_bbox_gt is not None and (face_bbox_gt.dim() != 2 or face_bbox_gt.size(-1) != 4):
        face_bbox_gt = None

    return BatchPack(bbox_gt=bbox_dense, label=label_dense, padding_mask=padding_mask, face_bbox_gt=face_bbox_gt)


# -------------------------
# visualization (with elem_id labels)
# -------------------------

def draw_layout_png(
    save_path: str,
    pred_xyxy: np.ndarray,         # [N,4]
    cls_idx: np.ndarray,           # [N]
    valid: np.ndarray,             # [N] bool
    face_id: int,
    skip_draw: Set[int],
    title: str = "",
    draw_ids: bool = False,
    id_with_class: bool = False,
    id_with_name: bool = False,
    id_pos: str = "topleft",
    gt_xyxy: Optional[np.ndarray] = None,
) -> None:
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    # GT (grey dashed)
    if gt_xyxy is not None:
        for elem_id in range(gt_xyxy.shape[0]):
            if not bool(valid[elem_id]):
                continue
            c = int(cls_idx[elem_id])
            if c in skip_draw:
                continue
            x0, y0, x1, y1 = gt_xyxy[elem_id]
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=1.0,
                                   edgecolor="gray", linestyle="--"))

    # Pred (blue/red)
    for elem_id in range(pred_xyxy.shape[0]):
        if not bool(valid[elem_id]):
            continue

        c = int(cls_idx[elem_id])
        if c in skip_draw:
            continue

        x0, y0, x1, y1 = pred_xyxy[elem_id]
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)

        is_face = (c == int(face_id))
        edge = "red" if is_face else "blue"
        lw = 2.2 if is_face else 1.2
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=lw, edgecolor=edge))

        # ✅ draw id text (token index)
        if draw_ids:
            txt = f"{elem_id}"
            if id_with_class:
                txt = f"{elem_id}:{c}"
            if id_with_name:
                name = CLASS_NAMES.get(c, str(c))
                txt = f"{txt}({name})"

            if id_pos == "center":
                tx = (x0 + x1) / 2.0
                ty = (y0 + y1) / 2.0
                ha, va = "center", "center"
            else:  # topleft
                tx = x0
                ty = y0
                ha, va = "left", "top"

            ax.text(
                tx, ty, txt,
                fontsize=7,
                color=edge,
                ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=edge, lw=0.6, alpha=0.85),
            )

    if title:
        ax.set_title(title, fontsize=9)

    fig.tight_layout(pad=0.2)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--dataset", type=str, default="crello_mainpart_face",
                        choices=["crello", "crello_mainpart", "crello_mainpart_face"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--variant", type=str, default="default")

    # dataset override (optional)
    parser.add_argument("--dataset_module", type=str, default="",
                        help="If set, import dataset from this module path (e.g., data.dataset.crello).")
    parser.add_argument("--dataset_class", type=str, default="",
                        help="If set, use this dataset class name (default: CrelloDataset).")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--start_idx", type=int, default=0)

    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--G_d_model", type=int, default=256)
    parser.add_argument("--G_nhead", type=int, default=4)
    parser.add_argument("--G_num_layers", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--bbox_format", type=str, default="cxcywh", choices=["cxcywh", "xywh", "xyxy"])
    parser.add_argument("--gt_bbox_format", type=str, default="cxcywh", choices=["cxcywh", "xywh", "xyxy"])
    parser.add_argument("--face_id", type=int, default=5)

    # mask flags:
    # default behavior: padding_mask=True means PAD
    # --mask_true_is_valid: padding_mask=True means VALID
    # --mask_is_padding_true: alias (same as default)
    parser.add_argument("--mask_true_is_valid", action="store_true",
                        help="If set, padding_mask=True means VALID.")
    parser.add_argument("--mask_is_padding_true", action="store_true",
                        help="(alias) padding_mask=True means PAD (default).")

    # generator import
    parser.add_argument("--G_module", type=str, default="model.layoutganpp")
    parser.add_argument("--G_class", type=str, default="")

    # draw / metrics
    parser.add_argument("--skip_draw_ids", type=str, default="3,4",
                        help="Comma-separated class ids NOT to draw (default skips background & mask).")
    parser.add_argument("--skip_overlap_ids", type=str, default="3,4",
                        help="Comma-separated class ids excluded from overlap metric.")
    parser.add_argument("--overlap_iou_thresh", type=float, default=0.1)
    parser.add_argument("--draw_gt", action="store_true",
                        help="Overlay GT boxes (grey dashed) for comparison.")

    # ID overlay
    parser.add_argument("--draw_ids", action="store_true", help="Draw token id on each bbox.")
    parser.add_argument("--id_with_class", action="store_true", help="Show id with class id (e.g., 12:2).")
    parser.add_argument("--id_with_name", action="store_true", help="Also show class name (e.g., 12:2(ImageElement)).")
    parser.add_argument("--id_pos", type=str, default="topleft", choices=["topleft", "center"])

    # num_classes override
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Fallback num_classes if cannot infer from ckpt.")

    args = parser.parse_args()

    if not HAS_PYG:
        raise RuntimeError("torch_geometric not available. Install torch_geometric in this env.")

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Ensure repo root on sys.path (helps imports)
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Dataset
    ds = build_dataset(args)
    sample0 = ds[0]
    if not isinstance(sample0, GeoData):
        raise RuntimeError(f"Expected dataset to return torch_geometric.data.Data, got {type(sample0)}")

    def pyg_collate(data_list):
        return GeoBatch.from_data_list(data_list)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=pyg_collate,
    )

    # Checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_prefix_if_present(sd, "module.")
    sd = strip_prefix_if_present(sd, "netG.")
    sd = strip_prefix_if_present(sd, "G.")

    # num_classes
    inferred_nc = infer_num_classes_from_ckpt(sd)
    num_classes = inferred_nc if inferred_nc is not None else args.num_classes
    if inferred_nc is None:
        print(f"[warn] cannot infer num_classes from ckpt; using --num_classes {num_classes}")

    # Generator
    netG = build_generator(args=args, num_classes=num_classes, sd=sd)
    try:
        missing, unexpected = netG.load_state_dict(sd, strict=False)
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    except RuntimeError as e:
        raise RuntimeError(
            "load_state_dict failed (shape mismatch). "
            "Likely --G_d_model/--G_nhead/--G_num_layers/--latent_size mismatch.\n"
            f"{e}"
        )

    netG = netG.to(device).eval()

    skip_draw = parse_int_set(args.skip_draw_ids)
    skip_overlap = parse_int_set(args.skip_overlap_ids)

    # mask conversion
    def padmask_to_valid(padding_mask: torch.Tensor) -> torch.Tensor:
        if args.mask_true_is_valid and args.mask_is_padding_true:
            raise ValueError("Conflicting flags: --mask_true_is_valid and --mask_is_padding_true")
        if args.mask_true_is_valid:
            return padding_mask
        return ~padding_mask  # default: True=PAD

    stat = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "bbox_format": args.bbox_format,
        "gt_bbox_format": args.gt_bbox_format,
        "face_id": args.face_id,
        "num_classes": num_classes,
        "num_samples": args.num_samples,
        "skip_draw_ids": sorted(list(skip_draw)),
        "skip_overlap_ids": sorted(list(skip_overlap)),
        "samples": [],
    }

    saved = 0

    with torch.no_grad():
        for bi, raw in enumerate(loader):
            if bi * args.batch_size < args.start_idx:
                continue

            pack = parse_batch(raw)

            label = pack.label.to(device)
            padding_mask = pack.padding_mask.to(device)   # True=PAD (dataset)
            valid = padmask_to_valid(padding_mask)        # True=VALID (for our filtering)

            # label -> indices Long for nn.Embedding
            if label.dim() == 3:
                label_idx = label.argmax(dim=-1).long()
            elif label.dim() == 2:
                label_idx = label.long()
            else:
                raise ValueError(f"Unsupported label shape: {tuple(label.shape)}")

            # pad positions safe id=0
            label_idx = label_idx.masked_fill(padding_mask, 0)

            # optional range sanity check (only on valid tokens)
            max_label = int(label_idx[valid].max().item()) if valid.any() else 0
            if max_label >= num_classes:
                print(f"[warn] label index out of range: max={max_label} >= num_classes={num_classes}")

            B, N = label_idx.shape
            z = torch.randn((B, N, args.latent_size), device=device)

            # forward (try common signatures)
            try:
                bbox_pred = netG(z, label_idx, padding_mask)
            except TypeError:
                try:
                    bbox_pred = netG(z, label_idx, valid)
                except TypeError:
                    bbox_pred = netG(z, label_idx)

            if not torch.isfinite(bbox_pred).all():
                raise RuntimeError("Found NaN/Inf in bbox_pred. Checkpoint likely diverged.")

            pred_xyxy = bbox_to_xyxy(bbox_pred, args.bbox_format)  # [B,N,4]
            cls_idx = label_idx  # [B,N]

            gt_xyxy = None
            if args.draw_gt and pack.bbox_gt is not None:
                gt_xyxy = bbox_to_xyxy(pack.bbox_gt.to(device), args.gt_bbox_format)

            for b_i in range(B):
                if saved >= args.num_samples:
                    break

                valid_i = valid[b_i]             # True=VALID
                pad_i = padding_mask[b_i]        # True=PAD
                cls_i = cls_idx[b_i]
                pred_i = pred_xyxy[b_i]

                # out-of-range ratio (valid tokens only; but your model seems in-range)
                denom = float(valid_i.float().sum().item()) + 1e-8
                oor = (((bbox_pred[b_i] < -0.05) | (bbox_pred[b_i] > 1.05)).float().sum().item()) / (denom * 4.0)

                # face overlap metric (exclude skip_overlap ids)
                use_i = valid_i & (~torch.isin(cls_i, torch.tensor(list(skip_overlap), device=cls_i.device)))
                face_mask = (cls_i == args.face_id) & use_i
                other_mask = (cls_i != args.face_id) & use_i

                face_boxes = pred_i[face_mask]
                other_boxes = pred_i[other_mask]

                max_iou = 0.0
                if face_boxes.numel() > 0 and other_boxes.numel() > 0:
                    ious = box_iou_xyxy(other_boxes[:, None, :], face_boxes[None, :, :])
                    max_iou = float(ious.max().item())

                flagged = (max_iou > args.overlap_iou_thresh)

                img_path = os.path.join(args.out_dir, f"sample_{saved:05d}.png")
                title = f"idx={saved} valid={int(valid_i.sum().item())} face={int(face_mask.sum().item())} " \
                        f"maxIoU={max_iou:.3f} oor={oor:.3f}"

                draw_layout_png(
                    save_path=img_path,
                    pred_xyxy=to_numpy(clamp01(pred_i)),
                    cls_idx=to_numpy(cls_i),
                    valid=to_numpy(valid_i),
                    face_id=args.face_id,
                    skip_draw=skip_draw,
                    title=title,
                    draw_ids=args.draw_ids,
                    id_with_class=args.id_with_class,
                    id_with_name=args.id_with_name,
                    id_pos=args.id_pos,
                    gt_xyxy=(to_numpy(clamp01(gt_xyxy[b_i])) if gt_xyxy is not None else None),
                )

                stat["samples"].append({
                    "sample_idx": saved,
                    "png": os.path.basename(img_path),
                    "num_valid": int(valid_i.long().sum().item()),
                    "num_face": int(face_mask.long().sum().item()),
                    "max_iou_nonface_vs_face": max_iou,
                    "out_of_range_ratio": float(oor),
                    "flagged_overlap": bool(flagged),
                })

                saved += 1

            if saved >= args.num_samples:
                break

    # summary
    if stat["samples"]:
        max_ious = [s["max_iou_nonface_vs_face"] for s in stat["samples"]]
        oors = [s["out_of_range_ratio"] for s in stat["samples"]]
        flags = [1.0 if s["flagged_overlap"] else 0.0 for s in stat["samples"]]
        stat["summary"] = {
            "saved": saved,
            "mean_max_iou": float(np.mean(max_ious)),
            "max_max_iou": float(np.max(max_ious)),
            "mean_out_of_range_ratio": float(np.mean(oors)),
            "flagged_overlap_ratio": float(np.mean(flags)),
        }
    else:
        stat["summary"] = {"saved": 0}

    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stat, f, indent=2, ensure_ascii=False)

    print(f"[Done] saved {saved} -> {args.out_dir}")
    print(f"[Done] metrics -> {out_json}")
    print(f"[Summary] {stat['summary']}")


if __name__ == "__main__":
    main()
