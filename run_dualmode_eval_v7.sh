#!/usr/bin/env bash
# 用法：
#   bash /home/albee/const_layout_whitespace/run_dualmode_eval_v6_yolo.sh
#
# 說明：
#   這版是在 v6 dual-mode 基礎上加入 YOLOv8n-face 推論前處理。
#   流程：
#   1. 使用者圖片放在 user_images/
#   2. infer_official_face_v14.py 使用 YOLOv8n-face 偵測人臉位置
#   3. 將偵測到的 face box 加入 layout 推論
#   4. raw / final 都會跑
#
# 注意：
#   - INFER 需要使用支援 YOLO 參數的版本，例如 infer_official_face_v14.py
#   - 需要有 yolov8n-face.pt
#   - IMAGE_PATH 要改成你的實際圖片檔名

set -euo pipefail

ROOT=/home/albee/const_layout_whitespace

# 使用加入 YOLO face detection 的 infer 程式
INFER=${ROOT}/infer_official_face_v14.py

# whitespace metric
WS=${ROOT}/whitespace_metric_v10.py

# 輸出資料夾
OUT_ROOT=${ROOT}/final_eval/official_v15_yolo

# =========================
# Basic inference settings
# =========================
N=3
K_FINAL=64
K_RAW=1
SEED=123
MAX_FG_AREA=0.33
MAX_ELEMS=5
ADD_FACE=1
SVG_SMALL_PROB=0.72

# =========================
# YOLO face detection settings
# =========================
# 這裡改成你放進 user_images 的圖片
IMAGE_PATH=${ROOT}/user_images/image01.png

# YOLOv8n-face 權重
FACE_MODEL=${ROOT}/yolov8n-face.pt

# 人臉偵測信心門檻
FACE_CONF=0.3

# 單張海報最多使用幾張臉
# 你目前 max_elems=5，通常建議先用 1
MAX_DETECTED_FACES=1

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

# Experiments:
#   EXP          = metric / output experiment name
#   INFER_STYLE  = inference script style argument
#   TARGET_STYLE = whitespace metric target_style argument
#
# Notes:
#   - inference supports: right / top / hybrid / frame
#   - metric supports: frame / side / tb / hybrid aliases
#   - side 使用 right-style inference
#   - tb 使用 top-style inference

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

# 如果你後來真的有命名為 side_final.pth / tb_final.pth，則自動優先使用。
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
      --image_path "${IMAGE_PATH}" \
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
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
      --image_path "${IMAGE_PATH}" \
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
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
done#!/usr/bin/env bash
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
#   - face 由 ADD_FACE=1 + YOLO 偵測加入
#   - 若 CUSTOM_LABELS=(2 1 1 0)，最後元素為 image + text + text + SVG + face

set -euo pipefail

ROOT=/home/albee/const_layout_whitespace

# 使用加入 YOLO face detection + image aspect ratio 的版本
INFER=${ROOT}/infer_official_face_v14.py

# whitespace metric
WS=${ROOT}/whitespace_metric_v10.py

# 輸出資料夾
OUT_ROOT=${ROOT}/final_eval/official_v16_yolo_custom_labels

# =========================
# Basic inference settings
# =========================
N=200
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
# face 由 ADD_FACE=1 和 YOLO 偵測加入。
#
# 這組代表：
# image + text + text + SVG + face
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
IMAGE_PATH=${ROOT}/user_images/test.jpg

# YOLOv8n-face 權重
FACE_MODEL=${ROOT}/yolov8n-face.pt

# 人臉偵測信心門檻
FACE_CONF=0.3

# 單張海報最多使用幾張臉
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