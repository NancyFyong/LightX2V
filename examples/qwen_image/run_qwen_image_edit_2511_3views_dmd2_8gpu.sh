#!/usr/bin/env bash
# Train Qwen-Image-Edit-2511 three-view DMD2 on eight GPUs.
#
# This is a production launcher: it uses 1024x1024 images, FSDP2, a frozen
# three-view SFT teacher, four-step DMD2 rollout matching, and a 2,000-iteration
# schedule. It resumes automatically from the configured output directory.

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
TRAIN_ROOT="${REPO_ROOT}/lightx2v_train"
CONFIG="${TRAIN_ROOT}/configs/train/dmd/qwen_image_edit_2511_dmd2_lora.yaml"

VENV="${VENV:-/workspace/user_code/40173/uv_venv/lightx2v}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export QWEN_IMAGE_EDIT_3VIEWS_MODEL="${QWEN_IMAGE_EDIT_3VIEWS_MODEL:-/workspace/user_code/40173/models/Qwen/Qwen-Image-Edit-2511-3views}"
export QWEN_IMAGE_EDIT_3VIEWS_TRAIN_JSONL="${QWEN_IMAGE_EDIT_3VIEWS_TRAIN_JSONL:-/workspace/user_code/40173/DiffSynth-Studio/dataset/imgs2uv_train/lightx2v_train.jsonl}"
export QWEN_IMAGE_EDIT_3VIEWS_VAL_JSONL="${QWEN_IMAGE_EDIT_3VIEWS_VAL_JSONL:-/workspace/user_code/40173/DiffSynth-Studio/dataset/imgs2uv_train/lightx2v_val.jsonl}"
export QWEN_IMAGE_EDIT_DMD2_OUTPUT_DIR="${QWEN_IMAGE_EDIT_DMD2_OUTPUT_DIR:-/workspace/user_code/40173/lightx2v_outputs/qwen_image_edit_2511_3views_dmd2_1024}"
export QWEN_IMAGE_EDIT_DMD2_INFER_DIR="${QWEN_IMAGE_EDIT_DMD2_INFER_DIR:-${QWEN_IMAGE_EDIT_DMD2_OUTPUT_DIR}/infer}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ ! -x "${VENV}/bin/torchrun" ]]; then
    echo "Missing torchrun: ${VENV}/bin/torchrun" >&2
    exit 1
fi

for required_path in \
    "${CONFIG}" \
    "${QWEN_IMAGE_EDIT_3VIEWS_MODEL}" \
    "${QWEN_IMAGE_EDIT_3VIEWS_TRAIN_JSONL}" \
    "${QWEN_IMAGE_EDIT_3VIEWS_VAL_JSONL}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne 8 ]]; then
    echo "This launcher requires exactly eight GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

mkdir -p "${QWEN_IMAGE_EDIT_DMD2_OUTPUT_DIR}" "${QWEN_IMAGE_EDIT_DMD2_INFER_DIR}"

cat <<EOF
Starting Qwen-Image-Edit-2511 three-view DMD2 training
  config:       ${CONFIG}
  GPUs:         ${CUDA_VISIBLE_DEVICES}
  model:        ${QWEN_IMAGE_EDIT_3VIEWS_MODEL}
  train JSONL:  ${QWEN_IMAGE_EDIT_3VIEWS_TRAIN_JSONL}
  val JSONL:    ${QWEN_IMAGE_EDIT_3VIEWS_VAL_JSONL}
  output:       ${QWEN_IMAGE_EDIT_DMD2_OUTPUT_DIR}
  resolution:   1024x1024
  iterations:   2000
EOF

cd "${TRAIN_ROOT}"
exec "${VENV}/bin/torchrun" \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --config "${CONFIG}"
