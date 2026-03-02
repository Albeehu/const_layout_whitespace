import argparse
import torch
import numpy as np

SVG_ID=0
TEXT_ID=1
IMG_ID=2
BG_ID=3
MASK_ID=4
FACE_ID=5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=str, default='fixed_sample.pt')
    ap.add_argument('--out', type=str, default='fixed_sample_v46.pt')
    ap.add_argument('--max_nodes', type=int, default=4)
    ap.add_argument('--max_faces', type=int, default=4)
    ap.add_argument('--num_classes', type=int, default=6)
    ap.add_argument('--latent_size', type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.src, map_location='cpu', weights_only=False)
    label = ck['label']
    mask = ck['mask'].bool()
    z = ck.get('z', None)
    bbox_real = ck.get('bbox_real', None)

    if label.dtype.is_floating_point:
        label = label.round().long()
    else:
        label = label.long()

    B, N = label.shape
    total_cap = args.max_nodes + args.max_faces

    # Prepare outputs
    out_label = torch.full((B, total_cap), args.num_classes - 1, dtype=torch.long)
    out_mask  = torch.zeros((B, total_cap), dtype=torch.bool)

    if z is not None:
        z = z.float()
        latent_size = z.size(-1)
    else:
        latent_size = args.latent_size

    out_z = torch.randn((B, total_cap, latent_size), dtype=torch.float32)

    # bbox_real is optional; used only for area-based selection if available.
    if bbox_real is not None:
        bbox_real = bbox_real.float()

    for b in range(B):
        valid_idx = torch.nonzero(mask[b], as_tuple=False).flatten().tolist()
        if not valid_idx:
            continue

        lab = label[b, valid_idx].cpu().numpy()

        # optional: use bbox_real to compute area and do the same SVG->IMG recode as v40
        if bbox_real is not None:
            bb = bbox_real[b, valid_idx].cpu().numpy()  # (Nv,4) cxcywh
            area = bb[:, 2] * bb[:, 3]
            lab = lab.copy()
            lab[(lab == SVG_ID) & (area > 0.15)] = IMG_ID
        else:
            bb = None
            area = None

        # split face / others
        face_idx_local = np.where(lab == FACE_ID)[0]
        other_idx_local = np.where(lab != FACE_ID)[0]

        # select others up to max_nodes using v40 priority + area (if available)
        if other_idx_local.size > 0:
            if bb is not None:
                other_area = area[other_idx_local]
            else:
                other_area = np.zeros(other_idx_local.shape[0], dtype=np.float32)
            other_lab = lab[other_idx_local]
            idxs = np.arange(other_idx_local.size)
            priority = np.ones(other_idx_local.size, dtype=np.int64)
            priority[other_lab == IMG_ID] = 0
            priority[other_lab == TEXT_ID] = 1
            priority[other_lab == SVG_ID] = 2
            priority[other_lab == BG_ID] = 3
            priority[other_lab == MASK_ID] = 4
            order = np.lexsort((idxs, -other_area, priority))
            keep_other_local = np.sort(order[:args.max_nodes])
            other_keep = other_idx_local[keep_other_local]
        else:
            other_keep = np.array([], dtype=np.int64)

        # select faces up to max_faces in original order
        face_keep = face_idx_local[:args.max_faces]

        # map back to original token indices in [0..N)
        keep_local = np.concatenate([other_keep, face_keep]).astype(np.int64)
        keep_orig = [valid_idx[i] for i in keep_local.tolist()]

        m = min(len(keep_orig), total_cap)
        if m == 0:
            continue

        out_label[b, :m] = torch.from_numpy(lab[keep_local][:m]).long()
        out_mask[b, :m] = True

        if z is not None:
            out_z[b, :m] = z[b, keep_orig[:m]]

    torch.save({'label': out_label, 'mask': out_mask, 'z': out_z}, args.out)

    # quick report
    face_tokens = int(((out_label == FACE_ID) & out_mask).sum().item())
    bc = torch.bincount(out_label[out_mask].reshape(-1), minlength=6).tolist()
    print('saved:', args.out)
    print(' label:', tuple(out_label.shape), out_label.dtype)
    print(' mask :', tuple(out_mask.shape), out_mask.dtype)
    print(' z    :', tuple(out_z.shape), out_z.dtype)
    print(' bincount(0..5):', bc)
    print(' face_tokens:', face_tokens)


if __name__ == '__main__':
    main()
