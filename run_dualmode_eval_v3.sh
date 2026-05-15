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

FRAME_BAND_MARGIN=0.10
FRAME_TARGET_W=0.28
FRAME_TARGET_H=0.38
FRAME_MAX_AR=1.8

STYLE=frame
MODE=raw

PRETRAIN_CKPT=${ROOT}/pretrained/layoutnet_crello.pth.tar
MY_CKPT=${ROOT}/final_eval/ckpts/frame_final.pth
# 或：
# MY_CKPT=/mnt/data/frame_final.pth

EXTRA_ARGS=(
  --frame_band_margin "${FRAME_BAND_MARGIN}"
  --frame_target_w "${FRAME_TARGET_W}"
  --frame_target_h "${FRAME_TARGET_H}"
  --frame_max_ar "${FRAME_MAX_AR}"
)

python "${INFER}" \
  --resume_ckpt "${PRETRAIN_CKPT}" \
  --out_dir "${OUT_ROOT}/${STYLE}_pretrain_raw" \
  --style "${STYLE}" \
  --n "${N}" \
  --k "${K}" \
  --seed "${SEED}" \
  --max_fg_area "${MAX_FG_AREA}" \
  --max_elems "${MAX_ELEMS}" \
  --add_face "${ADD_FACE}" \
  --svg_small_prob "${SVG_SMALL_PROB}" \
  --infer_mode "${MODE}" \
  --summary_name "pretrain_${STYLE}_raw" \
  "${EXTRA_ARGS[@]}"

python "${INFER}" \
  --resume_ckpt "${MY_CKPT}" \
  --out_dir "${OUT_ROOT}/${STYLE}_mine_raw" \
  --style "${STYLE}" \
  --n "${N}" \
  --k "${K}" \
  --seed "${SEED}" \
  --max_fg_area "${MAX_FG_AREA}" \
  --max_elems "${MAX_ELEMS}" \
  --add_face "${ADD_FACE}" \
  --svg_small_prob "${SVG_SMALL_PROB}" \
  --infer_mode "${MODE}" \
  --summary_name "mine_${STYLE}_raw" \
  "${EXTRA_ARGS[@]}"

python "${WS}" \
  --input "${OUT_ROOT}/${STYLE}_pretrain_raw/generated_layouts.pkl" \
  --output_csv "${OUT_ROOT}/${STYLE}_pretrain_raw/whitespace_metrics_allstyles.csv" \
  --summary_json "${OUT_ROOT}/${STYLE}_pretrain_raw/whitespace_summary_allstyles.json" \
  --summary_txt "${OUT_ROOT}/${STYLE}_pretrain_raw/whitespace_summary_allstyles.txt" \
  --box_format cxcywh \
  --ignore_labels 5 \
  --current_name "pretrain_${STYLE}_raw"

python "${WS}" \
  --input "${OUT_ROOT}/${STYLE}_mine_raw/generated_layouts.pkl" \
  --output_csv "${OUT_ROOT}/${STYLE}_mine_raw/whitespace_metrics_allstyles.csv" \
  --summary_json "${OUT_ROOT}/${STYLE}_mine_raw/whitespace_summary_allstyles.json" \
  --summary_txt "${OUT_ROOT}/${STYLE}_mine_raw/whitespace_summary_allstyles.txt" \
  --box_format cxcywh \
  --ignore_labels 5 \
  --current_name "mine_${STYLE}_raw" \
  --compare_summary_json "${OUT_ROOT}/${STYLE}_pretrain_raw/whitespace_summary_allstyles.json" \
  --compare_name "pretrain_${STYLE}_raw"

echo "Done. Check:"
echo "${OUT_ROOT}/${STYLE}_mine_raw/whitespace_summary_allstyles.txt"