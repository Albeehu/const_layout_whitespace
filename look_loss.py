#看loss_G loss_D
import os
from tensorboard.backend.event_processing import event_accumulator

def get_latest_loss(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    # 這裡的 'Loss/Generator' 必須對應你程式碼中 writer.add_scalar 的名稱
    try:
        tags = ea.Tags()['scalars']
        print(f"\n實驗目錄: {log_dir}")
        for tag in tags:
            if 'Loss' in tag or 'Score' in tag:
                events = ea.Scalars(tag)
                last_val = events[-1].value
                print(f"  - {tag:25s}: {last_val:.4f}")
    except Exception as e:
        print(f"讀取 {log_dir} 出錯: {e}")

# 填入你的實驗路徑
runs_root = "/home/albee/const_layout_whitespace/output/crello/LayoutGAN++"
experiments = ["crello_whitespace_pro_ignore", "crello_wr0.5ignore_v1"]

for exp in experiments:
    path = os.path.join(runs_root, exp)
    if os.path.exists(path):
        get_latest_loss(path)