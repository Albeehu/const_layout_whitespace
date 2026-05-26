#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/albee/const_layout_whitespace

# 這版只做 layout-only：
# - 支援多張 image（--image_paths）
# - 支援多個 SVG 資產（--svg_paths）
# - image 會等比例縮放
# - SVG 會保持原始比例，且每個 SVG box 的長邊固定大小
# - 支援 YOLO face detection，將 face 追加到各自 image box
# - 不需要文字輸入
# - 不需要背景圖
# - 不輸出 real poster，只輸出 layout 色塊圖與評估檔
#要自己放svg 等比例縮放

INFER=${ROOT}/infer_official_face_v22.py
WS=${ROOT}/whitespace_metric_v10.py
OUT_ROOT=${ROOT}/final_eval/official_v29_hollow_outline_strict_image_aspect

# =========================
# Basic inference settings
# =========================
N=5
K_FINAL=64
K_RAW=1
SEED=123
MAX_FG_AREA=0.33

# 這個數字要 >= (CUSTOM_LABELS 元素數量 + 預期 appended faces)
# 範例：2 image + 2 svg = 4 個元素
# 若每張 image 最多偵測 1 張 face，則 total = 4 + 2 = 6
MAX_ELEMS=15
ADD_FACE=1
SVG_SMALL_PROB=0.72

# =========================
# 指定生成元素
# 0 = SVG / vector
# 1 = text
# 2 = image
# 5 = face（不要手動寫進 CUSTOM_LABELS）
# =========================
CUSTOM_LABELS=(2 2 2 0 1 1 1)

# =========================
# Multi-image input
# 每一張圖片依序對應到一個 label=2 image box
# 若 image box 比圖片多，會循環使用圖片
# =========================
IMAGE_PATHS=(
  "${ROOT}/user_images/image1.png"
  "${ROOT}/user_images/image2.png"
  "${ROOT}/user_images/image3.png"
)

# =========================
# Multi-SVG input
# 每一個 SVG 路徑依序對應到一個 label=0 SVG box
# 若 SVG box 比 SVG 檔多，會循環使用 SVG
# 注意：這版不會真的把 SVG 畫到輸出圖，只會讀 SVG 長寬比，
# 用來讓 layout 裡的 SVG box 在最後輸出前保持嚴格等比例縮放。
# =========================
SVG_PATHS=(
  "${ROOT}/user_images/svg1.png"
 
)

# =========================
# YOLO face detection settings
# =========================
FACE_MODEL=${ROOT}/yolov8n-face.pt
FACE_CONF=0.3
MAX_DETECTED_FACES=1
PRESERVE_IMAGE_ASPECT=1
PRESERVE_SVG_ASPECT=1
SVG_ROLE_AFFECTS_SIZE=0
FIXED_SVG_LONG_SIDE=0.18
PREVIEW_W=80
PREVIEW_H=120
PREVIEW_BORDER_WIDTH=1
PREVIEW_FILL_ALPHA=0

# =========================
# Frame style settings
# =========================
FRAME_BAND_MARGIN=0.10
FRAME_TARGET_W=0.28
FRAME_TARGET_H=0.38
FRAME_MAX_AR=1.8

# =========================
# Checkpoints
# =========================
PRETRAIN_CKPT=${ROOT}/pretrained/layoutnet_crello.pth.tar

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

if [[ -f "${ROOT}/final_eval/ckpts/side_final.pth" ]]; then
  IMPROVED_CKPT[side]="${ROOT}/final_eval/ckpts/side_final.pth"
fi

if [[ -f "${ROOT}/final_eval/ckpts/tb_final.pth" ]]; then
  IMPROVED_CKPT[tb]="${ROOT}/final_eval/ckpts/tb_final.pth"
fi

# =========================
# Basic checks
# =========================
echo "INFER=${INFER}"
echo "WS=${WS}"
echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "IMAGE_PATHS=${IMAGE_PATHS[*]}"
echo "SVG_PATHS=${SVG_PATHS[*]}"
echo "FACE_MODEL=${FACE_MODEL}"
echo "CUSTOM_LABELS=${CUSTOM_LABELS[*]}"
echo "MAX_ELEMS=${MAX_ELEMS}"
echo "ADD_FACE=${ADD_FACE}"

test -f "${INFER}" || { echo "ERROR: INFER not found: ${INFER}"; exit 1; }
test -f "${WS}" || { echo "ERROR: WS not found: ${WS}"; exit 1; }
test -f "${PRETRAIN_CKPT}" || { echo "ERROR: PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}"; exit 1; }

for IMG in "${IMAGE_PATHS[@]}"; do
  test -f "${IMG}" || { echo "ERROR: IMAGE_PATH not found: ${IMG}"; exit 1; }
done

for SVG in "${SVG_PATHS[@]}"; do
  test -f "${SVG}" || { echo "ERROR: SVG_PATH not found: ${SVG}"; exit 1; }
done

if [[ ${#IMAGE_PATHS[@]} -gt 0 ]]; then
  test -f "${FACE_MODEL}" || { echo "ERROR: FACE_MODEL not found: ${FACE_MODEL}"; exit 1; }
fi

mkdir -p "${OUT_ROOT}"

# =========================
# Main loop
# =========================
for EXP in frame side tb hybrid; do
  STYLE="${INFER_STYLE[$EXP]}"
  TSTYLE="${TARGET_STYLE[$EXP]}"
  MY_CKPT="${IMPROVED_CKPT[$EXP]}"

  echo "========================================"
  echo "EXP=${EXP}"
  echo "  infer style   = ${STYLE}"
  echo "  target style  = ${TSTYLE}"
  echo "  pretrain ckpt = ${PRETRAIN_CKPT}"
  echo "  improved ckpt = ${MY_CKPT}"
  echo "  image paths   = ${IMAGE_PATHS[*]}"
  echo "  svg paths     = ${SVG_PATHS[*]}"
  echo "  face model    = ${FACE_MODEL}"
  echo "  labels        = ${CUSTOM_LABELS[*]}"
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

  for MODE in raw final; do
    if [[ "${MODE}" == "raw" ]]; then
      K=${K_RAW}
    else
      K=${K_FINAL}
    fi

    PRETRAIN_OUT="${OUT_ROOT}/${EXP}_pretrain_${MODE}"
    IMPROVED_OUT="${OUT_ROOT}/${EXP}_improved_${MODE}"

    echo "---------- ${EXP} / ${MODE} / pretrain ----------"
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
      --labels "${CUSTOM_LABELS[@]}" \
      --image_paths "${IMAGE_PATHS[@]}" \
      --svg_paths "${SVG_PATHS[@]}" \
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
      --preserve_image_aspect "${PRESERVE_IMAGE_ASPECT}" \
      --preserve_svg_aspect "${PRESERVE_SVG_ASPECT}" \
      --svg_role_affects_size "${SVG_ROLE_AFFECTS_SIZE}" \
      --fixed_svg_long_side "${FIXED_SVG_LONG_SIDE}" \
      --preview_w "${PREVIEW_W}" \
      --preview_h "${PREVIEW_H}" \
      --preview_border_width "${PREVIEW_BORDER_WIDTH}" \
      --preview_fill_alpha "${PREVIEW_FILL_ALPHA}" \
      "${EXTRA_ARGS[@]}"

    echo "---------- ${EXP} / ${MODE} / improved ----------"
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
      --labels "${CUSTOM_LABELS[@]}" \
      --image_paths "${IMAGE_PATHS[@]}" \
      --svg_paths "${SVG_PATHS[@]}" \
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
      --preserve_image_aspect "${PRESERVE_IMAGE_ASPECT}" \
      --preserve_svg_aspect "${PRESERVE_SVG_ASPECT}" \
      --svg_role_affects_size "${SVG_ROLE_AFFECTS_SIZE}" \
      --fixed_svg_long_side "${FIXED_SVG_LONG_SIDE}" \
      --preview_w "${PREVIEW_W}" \
      --preview_h "${PREVIEW_H}" \
      --preview_border_width "${PREVIEW_BORDER_WIDTH}" \
      --preview_fill_alpha "${PREVIEW_FILL_ALPHA}" \
      "${EXTRA_ARGS[@]}"

    echo "---------- ${EXP} / ${MODE} / metric pretrain ----------"
    python "${WS}" \
      --input "${PRETRAIN_OUT}/generated_layouts.pkl" \
      --output_csv "${PRETRAIN_OUT}/whitespace_metrics_allstyles.csv" \
      --summary_json "${PRETRAIN_OUT}/whitespace_summary_allstyles.json" \
      --summary_txt "${PRETRAIN_OUT}/whitespace_summary_allstyles.txt" \
      --box_format cxcywh \
      --ignore_labels 5 \
      --current_name "pretrain_${EXP}_${MODE}" \
      --target_style "${TSTYLE}"

    echo "---------- ${EXP} / ${MODE} / metric improved ----------"
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
done

echo "Done. Check summaries:"
for EXP in frame side tb hybrid; do
  echo "  ${OUT_ROOT}/${EXP}_improved_raw/whitespace_summary_allstyles.txt"
  echo "  ${OUT_ROOT}/${EXP}_improved_final/whitespace_summary_allstyles.txt"
done
