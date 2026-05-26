#!/usr/bin/env bash
# 用法：
#   bash /home/albee/const_layout_whitespace/run_dualmode_eval_v7.sh
#
# 說明：
#   這版是在 run_dualmode_eval_v6 的基礎上加入：
#   1. YOLOv8n-face 人臉偵測
#   2. 指定生成元素 CUSTOM_LABELS
#   3. 使用 infer_official_face_v14.py
#
# 元素 label：
#   0 = SVG / vector 裝飾圖
#   1 = text 文字
#   2 = image 圖片
#   5 = face 人臉
#
# 注意：
#   - CUSTOM_LABELS 裡面不要放 5
#   - face 由 ADD_FACE=1 + YOLO 偵測結果加入；沒有偵測到就不加 face
#   - 若 CUSTOM_LABELS=(2 1 1 0)，YOLO 偵測到 0 張臉 => image + text + text + SVG；偵測到 1 張臉 => image + text + text + SVG + face

set -euo pipefail

ROOT=/home/albee/const_layout_whitespace

# 使用加入 YOLO face detection + 固定 labels + detected-face-only 的版本
INFER=${ROOT}/infer_official_face_v16.py

# whitespace metric
WS=${ROOT}/whitespace_metric_v10.py

# 輸出資料夾
OUT_ROOT=${ROOT}/final_eval/official_v17_yolo_custom_detected_faces

# =========================
# Basic inference settings
# =========================
N=2
K_FINAL=64
K_RAW=1
SEED=123
MAX_FG_AREA=0.33
MAX_ELEMS=5
ADD_FACE=1
SVG_SMALL_PROB=0.72

# =========================
# 指定生成元素
# =========================
# 0 = SVG / vector
# 1 = text
# 2 = image
#
# 不要把 5 寫進 CUSTOM_LABELS。
# face 由 ADD_FACE=1 和 YOLO 偵測結果加入。
# 沒偵測到 face 就不加 face，不會補 synthetic face。
#
# 這組代表：
# image + text + text + SVG，若 YOLO 有偵測到臉才額外加 face
CUSTOM_LABELS=(2 1 1 0)

# 如果你想改成其他組合，可以改這裡：
# image + text + face:
# CUSTOM_LABELS=(2 1)
#
# image + text + text + face:
# CUSTOM_LABELS=(2 1 1)
#
# image + text + SVG + SVG + face:
# CUSTOM_LABELS=(2 1 0 0)

# =========================
# YOLO face detection settings
# =========================
# 改成你的實際圖片檔名
IMAGE_PATH=${ROOT}/user_images/image01.png

# YOLOv8n-face 權重
FACE_MODEL=${ROOT}/yolov8n-face.pt

# 人臉偵測信心門檻
FACE_CONF=0.3

# 單張海報最多使用幾張 YOLO 偵測到的人臉
# 如果偵測到 2 張臉且這裡設 2，最後會加 2 個 label=5。
# 注意：MAX_ELEMS 必須 >= CUSTOM_LABELS 數量 + 實際加入的人臉數。
MAX_DETECTED_FACES=1

# 是否保持使用者原圖長寬比
PRESERVE_IMAGE_ASPECT=1

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

# 如果你後來有 side_final.pth / tb_final.pth，則自動優先使用。
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
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "FACE_MODEL=${FACE_MODEL}"
echo "CUSTOM_LABELS=${CUSTOM_LABELS[*]}"
echo "MAX_ELEMS=${MAX_ELEMS}"
echo "ADD_FACE=${ADD_FACE}"

test -f "${INFER}" || { echo "ERROR: INFER not found: ${INFER}"; exit 1; }
test -f "${WS}" || { echo "ERROR: WS not found: ${WS}"; exit 1; }
test -f "${PRETRAIN_CKPT}" || { echo "ERROR: PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}"; exit 1; }
test -f "${IMAGE_PATH}" || { echo "ERROR: IMAGE_PATH not found: ${IMAGE_PATH}"; exit 1; }
test -f "${FACE_MODEL}" || { echo "ERROR: FACE_MODEL not found: ${FACE_MODEL}"; exit 1; }

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
  echo "  image path    = ${IMAGE_PATH}"
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
      --image_path "${IMAGE_PATH}" \
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
      --preserve_image_aspect "${PRESERVE_IMAGE_ASPECT}" \
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
      --image_path "${IMAGE_PATH}" \
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
      --preserve_image_aspect "${PRESERVE_IMAGE_ASPECT}" \
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