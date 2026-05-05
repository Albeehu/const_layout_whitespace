#!/usr/bin/env bash
#用法 bash /home/albee/const_layout_whitespace/run_dualmode_eval_v2.sh用來計算baseline improve的留白品質
set -euo pipefail

ROOT=/home/albee/const_layout_whitespace
INFER=/home/albee/const_layout_whitespace/infer_official_face_v6.py
WS=/home/albee/const_layout_whitespace/whitespace_metric_v6.py
OUT_ROOT=${ROOT}/final_eval/official_v7

N=200
K_FINAL=64
K_RAW=1
SEED=123
MAX_FG_AREA=0.33
MAX_ELEMS=5
ADD_FACE=1
SVG_SMALL_PROB=0.72

FRAME_BAND_MARGIN=0.10
FRAME_TARGET_W=0.28
FRAME_TARGET_H=0.38
FRAME_MAX_AR=1.8

declare -A IMPROVED_CKPT=(
  [right]="${ROOT}/final_eval/ckpts/right_final.pth"
  [hybrid]="${ROOT}/final_eval/ckpts/hybrid_final.pth"
  [top]="${ROOT}/final_eval/ckpts/top_final.pth"
  [frame]="${ROOT}/final_eval/ckpts/frame_final.pth"
)

declare -A BASELINE_CKPT=(
  [right]="${ROOT}/output/crello/LayoutGAN++/v56_right_baseline/checkpoints/ckpt_25500.pth"
  [hybrid]="${ROOT}/output/crello/LayoutGAN++/v56_hybrid_baseline/checkpoints/ckpt_25000.pth"
  [top]="${ROOT}/output/crello/LayoutGAN++/v56_top_baseline/checkpoints/ckpt_22000.pth"
  [frame]="${ROOT}/output/crello/LayoutGAN++/v57_frame_baseline/checkpoints/ckpt_45000.pth"
)

for STYLE in right hybrid top frame; do
  EXTRA_ARGS=()
  if [[ "${STYLE}" == "frame" ]]; then
    EXTRA_ARGS=(
      --frame_band_margin "${FRAME_BAND_MARGIN}"
      --frame_target_w "${FRAME_TARGET_W}"
      --frame_target_h "${FRAME_TARGET_H}"
      --frame_max_ar "${FRAME_MAX_AR}"
    )
  fi

  for MODE in raw final; do
    if [[ "${MODE}" == "raw" ]]; then
      K=${K_RAW}
    else
      K=${K_FINAL}
    fi

    python "${INFER}"       --resume_ckpt "${BASELINE_CKPT[$STYLE]}"       --out_dir "${OUT_ROOT}/${STYLE}_baseline_${MODE}"       --style "${STYLE}"       --n "${N}"       --k "${K}"       --seed "${SEED}"       --max_fg_area "${MAX_FG_AREA}"       --max_elems "${MAX_ELEMS}"       --add_face "${ADD_FACE}"       --svg_small_prob "${SVG_SMALL_PROB}"       --infer_mode "${MODE}"       --summary_name "baseline_${STYLE}_${MODE}"       "${EXTRA_ARGS[@]}"

    python "${INFER}"       --resume_ckpt "${IMPROVED_CKPT[$STYLE]}"       --out_dir "${OUT_ROOT}/${STYLE}_improved_${MODE}"       --style "${STYLE}"       --n "${N}"       --k "${K}"       --seed "${SEED}"       --max_fg_area "${MAX_FG_AREA}"       --max_elems "${MAX_ELEMS}"       --add_face "${ADD_FACE}"       --svg_small_prob "${SVG_SMALL_PROB}"       --infer_mode "${MODE}"       --summary_name "improved_${STYLE}_${MODE}"       "${EXTRA_ARGS[@]}"

    python "${WS}"       --input "${OUT_ROOT}/${STYLE}_baseline_${MODE}/generated_layouts.pkl"       --output_csv "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_metrics_allstyles.csv"       --summary_json "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_summary_allstyles.json"       --summary_txt "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_summary_allstyles.txt"       --box_format cxcywh       --ignore_labels 5       --current_name "baseline_${STYLE}_${MODE}"

    python "${WS}"       --input "${OUT_ROOT}/${STYLE}_improved_${MODE}/generated_layouts.pkl"       --output_csv "${OUT_ROOT}/${STYLE}_improved_${MODE}/whitespace_metrics_allstyles.csv"       --summary_json "${OUT_ROOT}/${STYLE}_improved_${MODE}/whitespace_summary_allstyles.json"       --summary_txt "${OUT_ROOT}/${STYLE}_improved_${MODE}/whitespace_summary_allstyles.txt"       --box_format cxcywh       --ignore_labels 5       --current_name "improved_${STYLE}_${MODE}"       --compare_summary_json "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_summary_allstyles.json"       --compare_name "baseline_${STYLE}_${MODE}"
  done
done

echo "Done. Check:"
echo "  ${OUT_ROOT}/right_improved_final/whitespace_summary_allstyles.txt"
echo "  ${OUT_ROOT}/hybrid_improved_final/whitespace_summary_allstyles.txt"
echo "  ${OUT_ROOT}/top_improved_final/whitespace_summary_allstyles.txt"
echo "  ${OUT_ROOT}/frame_improved_final/whitespace_summary_allstyles.txt"
echo "  ${OUT_ROOT}/right_improved_raw/whitespace_summary_allstyles.txt"
echo "  ${OUT_ROOT}/hybrid_improved_raw/whitespace_summary_allstyles.txt"
echo "  ${OUT_ROOT}/top_improved_raw/whitespace_summary_allstyles.txt"
echo "  ${OUT_ROOT}/frame_improved_raw/whitespace_summary_allstyles.txt"
