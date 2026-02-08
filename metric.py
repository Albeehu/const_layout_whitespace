import pickle
import numpy as np
import multiprocessing as mp
from itertools import chain
from scipy.optimize import linear_sum_assignment

import torch
from torch_geometric.utils import to_dense_adj
from pytorch_fid.fid_score import calculate_frechet_distance

from model.layoutnet import LayoutNet
from util import convert_xywh_to_ltrb
from data.util import RelSize, RelLoc, detect_size_relation, detect_loc_relation


class LayoutFID():
    def __init__(self, dataset_name, device='cpu'):
        num_label = 13 if dataset_name == 'rico' else 5
        self.model = LayoutNet(num_label).to(device)

        # load pre-trained LayoutNet
        tmpl = './pretrained/layoutnet_{}.pth.tar'
        # 🔹把新 dataset 名稱 map 回原本的權重名稱
        alias = {
            "crello_mainpart": "crello",
        }
        weight_name = alias.get(dataset_name, dataset_name)
        state_dict = torch.load(tmpl.format(weight_name), map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.requires_grad_(False)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, bbox, label, padding_mask, real=False):
        if real and type(self.real_features) != list:
            return

        feats = self.model.extract_features(bbox.detach(), label, padding_mask)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)


def compute_iou(box_1, box_2):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                   lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    return iou


def __compute_maximum_iou_for_layout(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([
        __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
        for i, j in zip(ii, jj)
    ]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list):
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(layouts_1, layouts_2, n_jobs=None):
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    # Debug: 看看有沒有交集
    print(
        "[compute_maximum_iou] len(keys_1) =", len(keys_1),
        "len(keys_2) =", len(keys_2),
        "len(intersection) =", len(keys),
    )

    #  完全沒有共同的 condition，直接回 NaN（或你想要的 0.0）
    if len(keys) == 0:
        return float("nan")

    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]

    # 如果沒指定 n_jobs，就用合理的值
    if n_jobs is None:
        n_jobs = min(len(args), mp.cpu_count() or 1)

    if len(args) == 0:
        return float("nan")

    with mp.Pool(n_jobs) as p:
        scores_list = p.map(__compute_maximum_iou, args)

    # 展平成一個 list
    flat_scores = list(chain.from_iterable(scores_list))

    if len(flat_scores) == 0:
        return float("nan")

    scores = np.asarray(flat_scores, dtype=np.float32)
    return float(scores.mean())


def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                     torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = torch.nan_to_num(ai / a1)

    return ar.sum(dim=(1, 2)) / mask.float().sum(-1)


def compute_alignment(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.permute(0, 3, 2, 1)
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)

    X = -torch.log(1 - X)

    return X.sum(-1) / mask.float().sum(-1)


def compute_violation(bbox_flatten, data):
    device = data.x.device
    failures, valid = [], []

    _zip = zip(data.edge_attr, data.edge_index.t())
    for gt, (i, j) in _zip:
        failure, _valid = 0, 0
        b1, b2 = bbox_flatten[i], bbox_flatten[j]

        # size relation
        if ~gt & 1 << RelSize.UNKNOWN:
            pred = detect_size_relation(b1, b2)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        # loc relation
        if ~gt & 1 << RelLoc.UNKNOWN:
            canvas = data.y[i].eq(0)
            pred = detect_loc_relation(b1, b2, canvas)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        failures.append(failure)
        valid.append(_valid)

    failures = torch.as_tensor(failures).to(device)
    failures = to_dense_adj(data.edge_index, data.batch, failures)
    valid = torch.as_tensor(valid).to(device)
    valid = to_dense_adj(data.edge_index, data.batch, valid)

    return failures.sum((1, 2)) / valid.sum((1, 2))

#新增FaceOcclusion
class FaceOcclusion():
    def __init__(self, face_pkl_path, device='cpu'):
        with open(face_pkl_path, 'rb') as f:
            self.face_data = pickle.load(f)
        self.device = device
        self.all_occlusion_rates = []

    def compute(self, pred_bboxes, index):
        """
        pred_bboxes: [N, 4] Tensor (xc, yc, w, h)
        index: 對應圖片的 index
        """
        faces = self.face_data[index]
        if not faces or pred_bboxes.numel() == 0:
            return 0.0

        # 1. 手動轉換座標，避開 util.py 可能發生的 unpack 錯誤
        # 假設 pred_bboxes 是 [N, 4]
        xc, yc, w, h = pred_bboxes[:, 0], pred_bboxes[:, 1], pred_bboxes[:, 2], pred_bboxes[:, 3]
        p_x1 = xc - w / 2
        p_y1 = yc - h / 2
        p_x2 = xc + w / 2
        p_y2 = yc + h / 2
        
        # [N, 4]
        pred_ltrb = torch.stack([p_x1, p_y1, p_x2, p_y2], dim=1).detach().cpu()
        
        img_occlusions = []
        for f_box in faces:
            # f_box [x, y, w, h] (左上角格式)
            fx1, fy1, fx2, fy2 = f_box[0], f_box[1], f_box[0]+f_box[2], f_box[1]+f_box[3]
            f_area = f_box[2] * f_box[3]
            if f_area <= 0: continue

            # 計算交集
            ix1 = torch.clamp(pred_ltrb[:, 0], min=fx1)
            iy1 = torch.clamp(pred_ltrb[:, 1], min=fy1)
            ix2 = torch.clamp(pred_ltrb[:, 2], max=fx2)
            iy2 = torch.clamp(pred_ltrb[:, 3], max=fy2)

            iw = torch.clamp(ix2 - ix1, min=0)
            ih = torch.clamp(iy2 - iy1, min=0)
            inter_area = iw * ih
            
            iof = inter_area / f_area
            
            # --- 重要修正：檢查 iof 是否為空 ---
            if iof.numel() > 0:
                max_iof = torch.max(iof).item()
            else:
                max_iof = 0.0
                
            img_occlusions.append(max_iof)
            self.all_occlusion_rates.append(max_iof)
            
        return max(img_occlusions) if img_occlusions else 0

    def report(self):
        """
        產出論文數據
        """
        if not self.all_occlusion_rates:
            return {"Avg_Occlusion": 0, "Frequency_gt_10": 0}
        
        avg_rate = np.mean(self.all_occlusion_rates)
        # 統計遮擋超過 10% 的嚴重案例比例
        freq = np.mean([1 if r > 0.1 else 0 for r in self.all_occlusion_rates])
        
        return {
            "Mean_Occlusion_Rate": avg_rate,
            "Occlusion_Frequency_10": freq
        }
