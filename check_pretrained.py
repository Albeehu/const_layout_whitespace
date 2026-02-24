#debug用 確認模型有吃到所有transformer權重層

import torch
import os
from pathlib import Path
from model.layoutganpp import Generator
from util import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_path = "/home/albee/const_layout_whitespace/pretrained/layoutnet_crello.pth.tar"
fixed_sample_path = "/home/albee/const_layout_whitespace/fixed_sample.pt"

def check():
    print("--- 正在執行【最終視覺化測試】---")
    out_dir = Path("check_results")
    out_dir.mkdir(exist_ok=True)

    # 這裡的參數必須這樣設，才能對齊那份 256 維度的權重
    netG = Generator(dim_latent=4, num_label=5, d_model=256, nhead=4, num_layers=4).to(device)
    g_dict = netG.state_dict()
    
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        ckpt_g = checkpoint['netG'] if 'netG' in checkpoint else checkpoint
        
        new_g_dict = {}
        for ck, cv in ckpt_g.items():
            clean_ck = ck.replace('module.', '').replace('netG.', '')
            tk = None
            
            # 映射翻譯表
            if 'emb_label.weight' in clean_ck: tk = 'emb_label.weight'
            elif 'fc_bbox.weight' in clean_ck: tk = 'fc_z.weight'    # [256, 4] -> 輸入層
            elif 'fc_bbox.bias' in clean_ck: tk = 'fc_z.bias'
            elif 'enc_fc_in.weight' in clean_ck: tk = 'fc_in.weight' # [256, 512] -> 拼接層
            elif 'enc_fc_in.bias' in clean_ck: tk = 'fc_in.bias'
            elif 'dec_transformer.layers' in clean_ck:
                tk = clean_ck.replace('dec_transformer.layers', 'transformer.layers')
            
            # 暴力搜尋座標輸出層 (形狀要是 [4, 256])
            if not tk:
                if cv.size() == torch.Size([4, 256]): tk = 'fc_out_bbox.weight'
                elif cv.size() == torch.Size([4]): tk = 'fc_out_bbox.bias'

            if tk and tk in g_dict and cv.size() == g_dict[tk].size():
                new_g_dict[tk] = cv

        print(f"成功對齊載入層數: {len(new_g_dict)} / {len(g_dict)}")
        netG.load_state_dict(new_g_dict, strict=False)

    # ... 推論部分 (記得 z 要擴充到 512) ...
    bbox_vis = None
    # --- 修正後的推論部分 ---
    try:
        ck = torch.load(fixed_sample_path, map_location=device)
        z = ck['z'][:5].to(device) # 原始 z, 不要補齊 512 了
        
        # 動態檢查：如果模型 fc_z 還是要 512 (雖然 55/55 應該是 4)
        expected_dim = netG.fc_z.in_features
        if z.shape[-1] != expected_dim:
            print(f"動態調整 z 維度: {z.shape[-1]} -> {expected_dim}")
            if z.shape[-1] < expected_dim:
                padding = torch.zeros((z.shape[0], z.shape[1], expected_dim - z.shape[-1]), device=device)
                z = torch.cat([z, padding], dim=-1)
            else:
                z = z[..., :expected_dim]

        label = torch.clamp(ck['label'][:5].to(device).long(), 0, 4)
        mask = ck['mask'][:5].to(device)

        netG.eval()
        with torch.no_grad():
            bbox_vis = netG(z, label, ~mask)
            bbox_vis = torch.clamp(bbox_vis, 0, 1)

        # 這裡定義繪圖顏色 (如果 util 沒定義)
        colors = [(128,128,128), (100,150,200), (128,0,128), (150,200,100), (220,220,220)]
        
        save_path = out_dir / "pretrained_check.png"
        # 修改這一行
        save_image(bbox_vis, label, mask, colors, str(save_path), 
           canvas_size=(600, 400), nrow=5)
        
        print(f"===> 【大功告成】真正的預訓練排版圖片已生成！")
        print(f"路徑在: {save_path.absolute()}")
        print("模型輸出的前三個座標樣本：\n", bbox_vis[0, :3, :])

    except Exception as e:
        print(f"推論再次失敗，原因: {e}")

if __name__ == "__main__":
    check()