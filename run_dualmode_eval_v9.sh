#!/usr/bin/env bash
# 用法：
#   bash /home/albee/const_layout_whitespace/run_dualmode_eval_v10_multi_image.sh
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
#   - 若 CUSTOM_LABELS=(2 2 1 1 0)，代表 layout 先生成 image + image + text + text + SVG
#   - 每張輸入圖片若 YOLO 有偵測到臉，才會在該圖片的 image box 內額外加入 face

set -euo pipefail

ROOT=/home/albee/const_layout_whitespace

# 使用加入 YOLO face detection + 固定 labels + detected-face-only 的版本
INFER=${ROOT}/infer_official_face_v18.py

# whitespace metric
WS=${ROOT}/whitespace_metric_v10.py

# 輸出資料夾
OUT_ROOT=${ROOT}/final_eval/official_v18_multi_image_render

# =========================
# Basic inference settings
# =========================
N=5
K_FINAL=64
K_RAW=1
SEED=123
MAX_FG_AREA=0.33
MAX_ELEMS=7
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
CUSTOM_LABELS=(2 0 1 1 1 0)

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
# 改成你的實際圖片檔名。v18 支援多張圖：每個 label=2 會依序使用一張圖。
# 例如 CUSTOM_LABELS=(2 2 1 1 0) 就代表一個 layout 有兩個 image box。
IMAGE_PATHS=(
  "${ROOT}/user_images/image01.png"
  "${ROOT}/user_images/image02.png"
)

# 保留單圖相容用；主要請用 IMAGE_PATHS。
IMAGE_PATH="${IMAGE_PATHS[0]}"

# YOLOv8n-face 權重
FACE_MODEL=${ROOT}/yolov8n-face.pt

# 人臉偵測信心門檻
FACE_CONF=0.3

# 每一張輸入圖片最多使用幾張 YOLO 偵測到的人臉。
# 注意：MAX_ELEMS 必須 >= CUSTOM_LABELS 數量 + 所有圖片實際加入的人臉總數。
# 例如 CUSTOM_LABELS=(2 2 1 1 0)，兩張圖各偵測 1 張臉 => 5 + 2 = 7，所以 MAX_ELEMS 至少 7。
MAX_DETECTED_FACES=1

# 是否保持使用者原圖長寬比
PRESERVE_IMAGE_ASPECT=1

# =========================
# Real asset rendering settings
# =========================
# 1=另外輸出 real_0000.png，把 image / SVG / text / background 都實際畫出來
RENDER_USER_IMAGE=1

# image_fit_mode:
# cover   = 等比例填滿 image box，可能裁切，海報常用
# contain = 完整顯示圖片，不裁切，但可能留白
IMAGE_FIT_MODE=cover

# SVG / vector 裝飾素材。可以放 .svg / .png / .jpg。
# 若使用 .svg，請先安裝：pip install cairosvg
# 多個 SVG 會依序放到 label=0 的位置；若 label=0 比素材多，會循環使用。
SVG_PATHS=(
  "${ROOT}/user_images/svg01.png"
)

# SVG 通常建議 contain，才不會被裁切或變形
SVG_FIT_MODE=contain

# 背景圖，可留空。背景通常建議 cover。
BACKGROUND_PATH=""
BG_FIT_MODE=cover

# 文字內容會依序放入 label=1 的 box
TEXT_VALUES=(
  "SALE"
  "New Collection"
)

RENDER_TEXT=1
FONT_PATH=""
TEXT_COLOR="#1E1E1E"
TEXT_BOX_ALPHA=0

# 輸出的真實海報尺寸
POSTER_W=1200
POSTER_H=800

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
echo "IMAGE_PATHS=${IMAGE_PATHS[*]}"
echo "FACE_MODEL=${FACE_MODEL}"
echo "CUSTOM_LABELS=${CUSTOM_LABELS[*]}"
echo "MAX_ELEMS=${MAX_ELEMS}"
echo "ADD_FACE=${ADD_FACE}"
echo "RENDER_USER_IMAGE=${RENDER_USER_IMAGE}"
echo "IMAGE_FIT_MODE=${IMAGE_FIT_MODE}"
echo "SVG_PATHS=${SVG_PATHS[*]}"
echo "SVG_FIT_MODE=${SVG_FIT_MODE}"
echo "BACKGROUND_PATH=${BACKGROUND_PATH}"
echo "BG_FIT_MODE=${BG_FIT_MODE}"
echo "TEXT_VALUES=${TEXT_VALUES[*]}"

test -f "${INFER}" || { echo "ERROR: INFER not found: ${INFER}"; exit 1; }
test -f "${WS}" || { echo "ERROR: WS not found: ${WS}"; exit 1; }
test -f "${PRETRAIN_CKPT}" || { echo "ERROR: PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}"; exit 1; }
for IMG in "${IMAGE_PATHS[@]}"; do
  test -f "${IMG}" || { echo "ERROR: image not found: ${IMG}"; exit 1; }
done
test -f "${FACE_MODEL}" || { echo "ERROR: FACE_MODEL not found: ${FACE_MODEL}"; exit 1; }

if [[ "${RENDER_USER_IMAGE}" == "1" ]]; then
  for SVG_PATH in "${SVG_PATHS[@]}"; do
    if [[ -n "${SVG_PATH}" ]]; then
      test -f "${SVG_PATH}" || { echo "ERROR: SVG_PATH not found: ${SVG_PATH}"; exit 1; }
    fi
  done

  if [[ -n "${BACKGROUND_PATH}" ]]; then
    test -f "${BACKGROUND_PATH}" || { echo "ERROR: BACKGROUND_PATH not found: ${BACKGROUND_PATH}"; exit 1; }
  fi

  if [[ -n "${FONT_PATH}" ]]; then
    test -f "${FONT_PATH}" || { echo "ERROR: FONT_PATH not found: ${FONT_PATH}"; exit 1; }
  fi
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
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
      --preserve_image_aspect "${PRESERVE_IMAGE_ASPECT}" \
      --render_user_image "${RENDER_USER_IMAGE}" \
      --image_fit_mode "${IMAGE_FIT_MODE}" \
      --svg_paths "${SVG_PATHS[@]}" \
      --svg_fit_mode "${SVG_FIT_MODE}" \
      --background_path "${BACKGROUND_PATH}" \
      --bg_fit_mode "${BG_FIT_MODE}" \
      --text_values "${TEXT_VALUES[@]}" \
      --render_text "${RENDER_TEXT}" \
      --font_path "${FONT_PATH}" \
      --text_color "${TEXT_COLOR}" \
      --text_box_alpha "${TEXT_BOX_ALPHA}" \
      --poster_w "${POSTER_W}" \
      --poster_h "${POSTER_H}" \
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
      --face_model "${FACE_MODEL}" \
      --face_conf "${FACE_CONF}" \
      --max_detected_faces "${MAX_DETECTED_FACES}" \
      --preserve_image_aspect "${PRESERVE_IMAGE_ASPECT}" \
      --render_user_image "${RENDER_USER_IMAGE}" \
      --image_fit_mode "${IMAGE_FIT_MODE}" \
      --svg_paths "${SVG_PATHS[@]}" \
      --svg_fit_mode "${SVG_FIT_MODE}" \
      --background_path "${BACKGROUND_PATH}" \
      --bg_fit_mode "${BG_FIT_MODE}" \
      --text_values "${TEXT_VALUES[@]}" \
      --render_text "${RENDER_TEXT}" \
      --font_path "${FONT_PATH}" \
      --text_color "${TEXT_COLOR}" \
      --text_box_alpha "${TEXT_BOX_ALPHA}" \
      --poster_w "${POSTER_W}" \
      --poster_h "${POSTER_H}" \
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