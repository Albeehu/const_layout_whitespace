#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/albee/const_layout_whitespace
INFER=${ROOT}/infer_official_face_v13.py
WS=${ROOT}/whitespace_metric_v8.py
OUT_ROOT=${ROOT}/final_eval/official_v14

N=200
K=1
SEED=123
MAX_FG_AREA=0.33
MAX_ELEMS=5
ADD_FACE=1
SVG_SMALL_PROB=0.72
MODE=raw

FRAME_BAND_MARGIN=0.10
FRAME_TARGET_W=0.28
FRAME_TARGET_H=0.38
FRAME_MAX_AR=1.8

PRETRAIN_CKPT=${ROOT}/pretrained/layoutnet_crello.pth.tar

# Experiments:
#   experiment name -> inference style -> metric target style -> improved ckpt
# Notes:
#   - inference script supports style: right / top / hybrid / frame
#   - metric target style supports: side / tb / hybrid / frame
#   - side uses right-style inference ckpt by default
#   - tb uses top-style inference ckpt by default

declare -A INFER_STYLE=(
  [frame]="frame"
  [side]="right"
  [tb]="top"
  [hybrid]="hybrid"
)

declare -A TARGET_STYLE=(
  [frame]="frame"
  [side]="side"
  [tb]="tb"
  [hybrid]="hybrid"
)

declare -A IMPROVED_CKPT=(
  [frame]="${ROOT}/final_eval/ckpts/frame_final.pth"
  [side]="${ROOT}/final_eval/ckpts/right_final.pth"
  [tb]="${ROOT}/final_eval/ckpts/top_final.pth"
  [hybrid]="${ROOT}/final_eval/ckpts/hybrid_final.pth"
)

# If you actually named ckpts side_final.pth / tb_final.pth, prefer those automatically.
if [[ -f "${ROOT}/final_eval/ckpts/side_final.pth" ]]; then
  IMPROVED_CKPT[side]="${ROOT}/final_eval/ckpts/side_final.pth"
fi
if [[ -f "${ROOT}/final_eval/ckpts/tb_final.pth" ]]; then
  IMPROVED_CKPT[tb]="${ROOT}/final_eval/ckpts/tb_final.pth"
fi

# Basic checks.
echo "INFER=${INFER}"
echo "WS=${WS}"
echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"

test -f "${INFER}" || { echo "ERROR: INFER not found: ${INFER}"; exit 1; }
test -f "${WS}" || { echo "ERROR: WS not found: ${WS}"; exit 1; }
test -f "${PRETRAIN_CKPT}" || { echo "ERROR: PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}"; exit 1; }

mkdir -p "${OUT_ROOT}"

for EXP in frame side tb hybrid; do
  STYLE="${INFER_STYLE[$EXP]}"
  TSTYLE="${TARGET_STYLE[$EXP]}"
  MY_CKPT="${IMPROVED_CKPT[$EXP]}"

  echo "========================================"
  echo "EXP=${EXP}"
  echo "  infer style   = ${STYLE}"
  echo "  target style  = ${TSTYLE}"
  echo "  improved ckpt = ${MY_CKPT}"
  echo "========================================"

  test -f "${MY_CKPT}" || { echo "ERROR: improved ckpt not found for ${EXP}: ${MY_CKPT}"; exit 1; }

  EXTRA_ARGS=()
  if [[ "${EXP}" == "frame" ]]; then
    EXTRA_ARGS=(
      --frame_band_margin "${FRAME_BAND_MARGIN}"
      --frame_target_w "${FRAME_TARGET_W}"
      --frame_target_h "${FRAME_TARGET_H}"
      --frame_max_ar "${FRAME_MAX_AR}"
    )
  fi

  PRETRAIN_OUT="${OUT_ROOT}/${EXP}_pretrain_${MODE}"
  IMPROVED_OUT="${OUT_ROOT}/${EXP}_improved_${MODE}"

  python "${INFER}" \
    --resume_ckpt "${PRETRAIN_CKPT}" \
    --out_dir "${PRETRAIN_OUT}" \
    --style "${STYLE}" \
    --n "${N}" \
    --k "${K}" \
    --seed "${SEED}" \
    --max_fg_area "${MAX_FG_AREA}" \
    --max_elems "${MAX_ELEMS}" \
    --add_face "${ADD_FACE}" \
    --svg_small_prob "${SVG_SMALL_PROB}" \
    --infer_mode "${MODE}" \
    --summary_name "pretrain_${EXP}_${MODE}" \
    "${EXTRA_ARGS[@]}"

  python "${INFER}" \
    --resume_ckpt "${MY_CKPT}" \
    --out_dir "${IMPROVED_OUT}" \
    --style "${STYLE}" \
    --n "${N}" \
    --k "${K}" \
    --seed "${SEED}" \
    --max_fg_area "${MAX_FG_AREA}" \
    --max_elems "${MAX_ELEMS}" \
    --add_face "${ADD_FACE}" \
    --svg_small_prob "${SVG_SMALL_PROB}" \
    --infer_mode "${MODE}" \
    --summary_name "improved_${EXP}_${MODE}" \
    "${EXTRA_ARGS[@]}"

  python "${WS}" \
    --input "${PRETRAIN_OUT}/generated_layouts.pkl" \
    --output_csv "${PRETRAIN_OUT}/whitespace_metrics_allstyles.csv" \
    --summary_json "${PRETRAIN_OUT}/whitespace_summary_allstyles.json" \
    --summary_txt "${PRETRAIN_OUT}/whitespace_summary_allstyles.txt" \
    --box_format cxcywh \
    --ignore_labels 5 \
    --current_name "pretrain_${EXP}_${MODE}" \
    --target_style "${TSTYLE}"

  python "${WS}" \
    --input "${IMPROVED_OUT}/generated_layouts.pkl" \
    --output_csv "${IMPROVED_OUT}/whitespace_metrics_allstyles.csv" \
    --summary_json "${IMPROVED_OUT}/whitespace_summary_allstyles.json" \
    --summary_txt "${IMPROVED_OUT}/whitespace_summary_allstyles.txt" \
    --box_format cxcywh \
    --ignore_labels 5 \
    --current_name "improved_${EXP}_${MODE}" \
    --target_style "${TSTYLE}" \
    --compare_summary_json "${PRETRAIN_OUT}/whitespace_summary_allstyles.json" \
    --compare_name "pretrain_${EXP}_${MODE}"

done

echo "Done. Check summaries:"
for EXP in frame side tb hybrid; do
  echo "  ${OUT_ROOT}/${EXP}_improved_${MODE}/whitespace_summary_allstyles.txt"
done
