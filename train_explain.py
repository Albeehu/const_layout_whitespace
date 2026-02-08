"""
================================== 變數名 ==================================
wr : whitespace ratio
face_coverage_loss:罰face被ele.遮到
pair_overlap_loss:罰ele.彼此overlap(可忽略face / background...label)

================================== Def. 內容 ==================================
1. wr >= wr_min 才要求留白好看
2. xywh_to_xyxy 把[cx, cy, w, h]變成[x1, y1,x2, y2](左, 上, 右, 下 邊界)
3. box_iou_xyxy
    input: boxes1 (N,4), boxes2 (N,4)
    若 boxes1 or 2 = null, return 0
    lt = max(左上角) = 交集左上角(N, M, 2)
    rb = max(右下角) = 交集右下角(N, M, 2)
    wh = 交集寬高, if < 0 會變 = 0
    inter = 交集面積
    area1, area2 算面積
    union = 聯集面積
    inter / union.clamp(1e-6) 算 IoU
4. Def. box_intersection_area_xyxy 算交集面積矩陣(K, M)
5. face_coverage_loss 算face被ele.遮的ratio, ratio大loss大
    lambda_face:這個loss的weight 0.3
    contain_thresh容器包含的門檻 0.95
    coverage(face) = sum_other inter_area(other, face) / area(face)
    inter.sum(dim=0)：對每個 face，把所有 other 的交集加起來 → 該 face 被遮的總面積
    coverage：把所有遮擋面積加總、除以 face 面積
    每張圖的 loss：平方後平均，再累積 ex.讓遮0.8比0.2懲罰大很多
6. pairwise_overlap_loss 元素overlap懲罰 ignore face
    return losses 平均（若空則回 0）
7. whitespace_style_loss 高留白會啟動的留白風格loss
    wr >= wr_min → 算四種留白風格分數 S_style，loss = 1 - S_style 
    wr_min = 0.7
    occ = area sum
    wr = 1 - occ
8. 四周留白風格：邊距越大、越平均越好
    S_frame = (mean_m - 0.5 * std_m).clamp(0.0, 1.0)
9. 左右留白風格 
    ***S_side = (h_max + 0.8 * (h_max - h_min) + 0.2 * v_mean).clamp(0.0, 1.0)
10. 上下留白風格 
    S_tb = (v_max + 0.8 * (v_max - v_min) + 0.2 * h_mean).clamp(0.0, 1.0)
11. 左右偏上下（角落留白）
    S_corner = torch.sqrt(S_side * S_tb).clamp(0.0, 1.0)
12. 四種風格喜好度 S_style
    S_style = torch.stack([S_frame, S_side, S_tb, S_corner]).max()
13. loss_style = 1 - S_style：分數越高 loss 越低
    loss_b = w_style * loss_style 乘上權重變成新的loss
    回傳 batch 平均（空則回 0）
14. add_argument
    基本訓練參數：實驗名、dataset、batch_size、iteration、seed
    General：latent_size、lr、aug_flip、num_workers
    logging/視覺化/驗證頻率 + fixed_sample：用固定 sample 跨 run 對齊可視化
    Generator 結構參數：d_model/nhead/num_layers
    L359–L365 Generator 結構參數：d_model/nhead/num_layers
    L367–L375: lambda_ws：留白風格 loss 權重
               wr_min：高留白門檻
               lambda_ov：overlap loss 權重
               lambda_face：face 遮擋 loss 權重
    L377–L383 Discriminator 結構參數
    L385–L387 parse args 並 print
15. Repro
    L396 out_dir = init_experiment(...)：建立實驗輸出目錄（通常含 log/checkpoint/images）
    ***L398 device：有 GPU 用 cuda:0 否則 CPU
16. load dataset
    ***L414–L421 val_dataset / val_dataloader：shuffle=False（穩定、也配合 fixed_sample）
    ***L426–L432 建 Generator 並 .to(device)
    L573–L577 若 crello_mainpart_face：
        把 face 當成 padding（不送進 FID）
        把 face label 改成 0（避免類別超出）
    
"""
