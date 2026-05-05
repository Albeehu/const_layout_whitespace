#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/albee/const_layout_whitespace
INFER=${ROOT}/infer_official_face_v6.py
WS=${ROOT}/whitespace_metric_v6.py

STYLE=top
CKPT_TAG=ckpt31750
RUN_TAG="top_compare_${CKPT_TAG}_$(date +%F_%H%M%S)"
OUT_ROOT=${ROOT}/final_eval/official_infer_dualmode_v5_tb_runs/${RUN_TAG}

mkdir -p "${OUT_ROOT}"

N=200
K_FINAL=64
K_RAW=1
SEED=123
MAX_FG_AREA=0.33
MAX_ELEMS=5
ADD_FACE=1
SVG_SMALL_PROB=0.72

BASELINE_CKPT="${ROOT}/output/crello/LayoutGAN++/v56_top_baseline/checkpoints/ckpt_22000.pth"
IMPROVED_CKPT="/home/albee/const_layout_whitespace/final_eval/ckpts/top_final.pth"

echo "Output will be saved to:"
echo "${OUT_ROOT}"

for MODE in raw final; do
  if [[ "${MODE}" == "raw" ]]; then
    K=${K_RAW}
  else
    K=${K_FINAL}
  fi

  echo "======================================"
  echo "Running TOP baseline ${MODE}, K=${K}"
  echo "======================================"

  python "${INFER}" \
    --resume_ckpt "${BASELINE_CKPT}" \
    --out_dir "${OUT_ROOT}/${STYLE}_baseline_${MODE}" \
    --style "${STYLE}" \
    --n "${N}" \
    --k "${K}" \
    --seed "${SEED}" \
    --max_fg_area "${MAX_FG_AREA}" \
    --max_elems "${MAX_ELEMS}" \
    --add_face "${ADD_FACE}" \
    --svg_small_prob "${SVG_SMALL_PROB}" \
    --infer_mode "${MODE}" \
    --summary_name "baseline_${STYLE}_${MODE}"

  echo "======================================"
  echo "Running TOP improved ${MODE}, K=${K}"
  echo "======================================"

  python "${INFER}" \
    --resume_ckpt "${IMPROVED_CKPT}" \
    --out_dir "${OUT_ROOT}/${STYLE}_improved_${MODE}" \
    --style "${STYLE}" \
    --n "${N}" \
    --k "${K}" \
    --seed "${SEED}" \
    --max_fg_area "${MAX_FG_AREA}" \
    --max_elems "${MAX_ELEMS}" \
    --add_face "${ADD_FACE}" \
    --svg_small_prob "${SVG_SMALL_PROB}" \
    --infer_mode "${MODE}" \
    --summary_name "improved_${STYLE}_${MODE}"

  echo "======================================"
  echo "Evaluating TOP baseline ${MODE}"
  echo "======================================"

  python "${WS}" \
    --input "${OUT_ROOT}/${STYLE}_baseline_${MODE}/generated_layouts.pkl" \
    --output_csv "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_metrics_allstyles.csv" \
    --summary_json "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_summary_allstyles.json" \
    --summary_txt "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_summary_allstyles.txt" \
    --box_format cxcywh \
    --ignore_labels 5 \
    --current_name "baseline_${STYLE}_${MODE}"

  echo "======================================"
  echo "Evaluating TOP improved ${MODE} and comparing with baseline"
  echo "======================================"

  python "${WS}" \
    --input "${OUT_ROOT}/${STYLE}_improved_${MODE}/generated_layouts.pkl" \
    --output_csv "${OUT_ROOT}/${STYLE}_improved_${MODE}/whitespace_metrics_allstyles.csv" \
    --summary_json "${OUT_ROOT}/${STYLE}_improved_${MODE}/whitespace_summary_allstyles.json" \
    --summary_txt "${OUT_ROOT}/${STYLE}_improved_${MODE}/whitespace_summary_allstyles.txt" \
    --box_format cxcywh \
    --ignore_labels 5 \
    --current_name "improved_${STYLE}_${MODE}" \
    --compare_summary_json "${OUT_ROOT}/${STYLE}_baseline_${MODE}/whitespace_summary_allstyles.json" \
    --compare_name "baseline_${STYLE}_${MODE}"
done

echo "Done."
echo "Check results:"
echo "${OUT_ROOT}/top_baseline_raw/whitespace_summary_allstyles.txt"
echo "${OUT_ROOT}/top_improved_raw/whitespace_summary_allstyles.txt"
echo "${OUT_ROOT}/top_baseline_final/whitespace_summary_allstyles.txt"
echo "${OUT_ROOT}/top_improved_final/whitespace_summary_allstyles.txt"
