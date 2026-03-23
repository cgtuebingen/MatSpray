#!/usr/bin/env bash

set -euo pipefail

# Defaults (change as needed)
METHOD_DIR_DEFAULT="${GS_METHOD_DIR:-/path/to/external/gaussian-splatting}"
DATA_DIR_DEFAULT="${GS_DATA_DIR:-/path/to/datasets/nerf_synthetic/lego}"
OUTPUT_ROOT_DEFAULT="${GS_OUTPUT_ROOT:-/path/to/results/gaussian_splatting}"
CUDA_DEVICES_DEFAULT="0"
ITERATIONS_DEFAULT="30000"

# Flags
DO_SETUP=false
DO_RENDER=false
EXTRA_ARGS=()

usage() {
	cat <<EOF
Usage: $0 [OPTIONS]

Run base Gaussian Splatting on a NeRF-format dataset (transforms*.json).

Options:
  --method-dir PATH        Path to gaussian-splatting repo
                           (default: ${METHOD_DIR_DEFAULT})
  --data-dir PATH          Path to NeRF dataset directory (with images/ and transforms*.json)
                           (default: ${DATA_DIR_DEFAULT})
  --output-root PATH       Root directory where outputs/models will be written
                           (default: ${OUTPUT_ROOT_DEFAULT})
  --iters N                Number of training iterations (default: ${ITERATIONS_DEFAULT})
  --cuda-devices LIST      CUDA_VISIBLE_DEVICES value (default: ${CUDA_DEVICES_DEFAULT})
  --render                 Also run rendering if render.py exists (default: disabled)
  --setup                  Install/compile dependencies for gaussian-splatting (one-time)
  --                       All following args are passed through to train.py (advanced)
  -h, --help               Show this help and exit

Examples:
  $0 \
    --data-dir "/path/to/datasets/nerf_synthetic/lego" \
    --output-root "/path/to/results/gaussian_splatting"

  # Passing extra args to train.py after --
  $0 -- --white_background
EOF
}

METHOD_DIR="${METHOD_DIR_DEFAULT}"
DATA_DIR="${DATA_DIR_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
ITERATIONS="${ITERATIONS_DEFAULT}"
CUDA_DEVICES="${CUDA_DEVICES_DEFAULT}"

# Parse arguments
while [[ $# -gt 0 ]]; do
	case "$1" in
		--method-dir)
			METHOD_DIR="$2"; shift 2 ;;
		--data-dir)
			DATA_DIR="$2"; shift 2 ;;
		--output-root)
			OUTPUT_ROOT="$2"; shift 2 ;;
		--iters)
			ITERATIONS="$2"; shift 2 ;;
		--cuda-devices)
			CUDA_DEVICES="$2"; shift 2 ;;
		--render)
			DO_RENDER=true; shift ;;
		--setup)
			DO_SETUP=true; shift ;;
		--)
			shift
			EXTRA_ARGS=("$@")
			break ;;
		-h|--help)
			usage; exit 0 ;;
		*)
			echo "Unknown option: $1" >&2
			usage; exit 1 ;;
	esac
done

# Sanity checks
if [[ ! -d "${METHOD_DIR}" ]]; then
	echo "ERROR: method dir not found: ${METHOD_DIR}" >&2
	exit 2
fi
if [[ ! -d "${DATA_DIR}" ]]; then
	echo "ERROR: data dir not found: ${DATA_DIR}" >&2
	exit 2
fi

# Detect dataset type (NeRF vs COLMAP) to warn appropriately
DATASET_TYPE="unknown"
if [[ -f "${DATA_DIR}/transforms.json" || -f "${DATA_DIR}/transforms_train.json" ]]; then
	DATASET_TYPE="nerf"
elif [[ -d "${DATA_DIR}/sparse" ]]; then
	DATASET_TYPE="colmap"
fi

if [[ "${DATASET_TYPE}" != "nerf" ]]; then
	echo "[warn] Dataset doesn't look like NeRF (no transforms*.json found). Proceeding anyway."
fi

# Derive scene name and output dir
SCENE_NAME=$(basename "${DATA_DIR}")
MODEL_DIR="${OUTPUT_ROOT}/${SCENE_NAME}"
mkdir -p "${MODEL_DIR}"

# GPU / CUDA env
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Info
echo "=== Gaussian Splatting (base, NeRF) ==="
echo "Method dir        : ${METHOD_DIR}"
echo "Data dir (NeRF)    : ${DATA_DIR}"
echo "Output root       : ${OUTPUT_ROOT}"
echo "Model dir         : ${MODEL_DIR}"
echo "Iterations        : ${ITERATIONS}"
echo "CUDA devices      : ${CUDA_VISIBLE_DEVICES}"
echo "Render after train: ${DO_RENDER}"

# Optional: quick GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi || true
fi

# Enter method dir
pushd "${METHOD_DIR}" >/dev/null

# Optional setup (one-time)
if [[ "${DO_SETUP}" == true ]]; then
	echo "[setup] Installing dependencies for gaussian-splatting..."
	if [[ -f requirements.txt ]]; then
		pip install -r requirements.txt
	fi
	# simple-knn submodule (common in base repo)
	if [[ -d submodules/simple-knn ]]; then
		pip install -e submodules/simple-knn
	fi
fi

# Verify entry points
if [[ ! -f train.py ]]; then
	echo "ERROR: train.py not found in ${METHOD_DIR}" >&2
	popd >/dev/null
	exit 3
fi

# Train

echo "[train] Starting training..."
_has_flag() {
    local flag="$1"; shift
    for v in "$@"; do
        if [[ "$v" == "$flag" ]]; then
            return 0
        fi
    done
    return 1
}

_find_free_port() {
    local base="${1:-6009}"; local range="${2:-1000}"; local end=$((base+range-1)); local p
    for ((p=base; p<=end; p++)); do
        if ! ss -ltn 2>/dev/null | grep -E -q "LISTEN\s+.*[:.]${p}\\b"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

cmd=(python train.py --source_path "${DATA_DIR}" --model_path "${MODEL_DIR}" --iterations "${ITERATIONS}")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
	cmd+=("${EXTRA_ARGS[@]}")
fi
if ! _has_flag "--disable_viewer" "${EXTRA_ARGS[@]}" && ! _has_flag "--port" "${EXTRA_ARGS[@]}"; then
    BASE_PORT="${GS_VIEWER_BASE_PORT:-6009}"
    if PORT=$(_find_free_port "${BASE_PORT}" 1000); then
        echo "[train] Auto-assigning viewer port: ${PORT}"
        cmd+=(--port "${PORT}")
    else
        echo "[warn] Could not find a free port in range ${BASE_PORT}..$((BASE_PORT+999)); proceeding without explicit --port"
    fi
fi
"${cmd[@]}"

echo "[train] Done. Check: ${MODEL_DIR}"

# Render (optional, only if render.py exists)
if [[ "${DO_RENDER}" == true ]]; then
	if [[ -f render.py ]]; then
		echo "[render] Rendering novel views..."
		python render.py \
			-m "${MODEL_DIR}" \
			-s "${DATA_DIR}" \
			-o "${MODEL_DIR}/renders"
		echo "[render] Outputs in: ${MODEL_DIR}/renders"
	else
		echo "[render] Skipped: render.py not found in ${METHOD_DIR}"
	fi
fi

popd >/dev/null

echo "All done." 