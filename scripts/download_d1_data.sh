#!/bin/bash
# =============================================================================
# Phase D1: Download datasets and checkpoints for multi-task validation
#
# Run this on-pod BEFORE run_d1_multitask.sh.
# Requires: gdown, zstd (pip install gdown; apt install zstd)
#
# Usage:
#   export STABLEWM_HOME=/workspace/data
#   bash scripts/download_d1_data.sh
# =============================================================================

set -euo pipefail

STABLEWM_HOME="${STABLEWM_HOME:-/workspace/data}"

echo "Downloading D1 multi-task data to ${STABLEWM_HOME}"
echo "=================================================="

# Ensure tools are available
pip install -q gdown
command -v zstd >/dev/null 2>&1 || apt-get install -y -qq zstd

# -----------------------------------------------------------------------------
# PushT — should already be on the volume from Phase 0-7
# -----------------------------------------------------------------------------
echo ""
echo "[1/4] PushT"
if [ -f "${STABLEWM_HOME}/pusht_expert_train.h5" ]; then
    echo "  Dataset: already present"
else
    echo "  Dataset: MISSING — download from Google Drive"
    echo "  TODO: gdown <PUSHT_DATASET_ID> -O ${STABLEWM_HOME}/pusht_expert_train.h5.zst"
    echo "        zstd -d ${STABLEWM_HOME}/pusht_expert_train.h5.zst"
fi

if [ -d "${STABLEWM_HOME}/pusht" ]; then
    echo "  Checkpoint: already present"
else
    echo "  Checkpoint: MISSING"
    echo "  TODO: gdown <PUSHT_CKPT_ID> -O ${STABLEWM_HOME}/pusht/lejepa_object.ckpt"
fi

# -----------------------------------------------------------------------------
# TwoRoom
# -----------------------------------------------------------------------------
echo ""
echo "[2/4] TwoRoom"
if [ -f "${STABLEWM_HOME}/tworoom.h5" ] || [ -f "${STABLEWM_HOME}/tworoom/tworoom.h5" ]; then
    echo "  Dataset: already present"
else
    echo "  Dataset: downloading..."
    # TODO: Replace with actual Google Drive file ID
    # gdown <TWOROOM_DATASET_ID> -O /tmp/tworoom.h5.zst
    # zstd -d /tmp/tworoom.h5.zst -o ${STABLEWM_HOME}/tworoom.h5
    echo "  TODO: Set TWOROOM_DATASET_ID and uncomment gdown commands"
fi

if [ -d "${STABLEWM_HOME}/tworoom" ] && ls "${STABLEWM_HOME}/tworoom/"*ckpt* >/dev/null 2>&1; then
    echo "  Checkpoint: already present"
else
    mkdir -p "${STABLEWM_HOME}/tworoom"
    echo "  Checkpoint: downloading..."
    # TODO: Replace with actual Google Drive file ID
    # gdown <TWOROOM_CKPT_ID> -O ${STABLEWM_HOME}/tworoom/lejepa_object.ckpt
    echo "  TODO: Set TWOROOM_CKPT_ID and uncomment gdown commands"
fi

# -----------------------------------------------------------------------------
# OGBench Cube
# -----------------------------------------------------------------------------
echo ""
echo "[3/4] OGBench Cube"
if [ -f "${STABLEWM_HOME}/ogbench/cube_single_expert.h5" ]; then
    echo "  Dataset: already present"
else
    mkdir -p "${STABLEWM_HOME}/ogbench"
    echo "  Dataset: downloading..."
    # TODO: Replace with actual Google Drive file ID
    # gdown <CUBE_DATASET_ID> -O /tmp/cube_single_expert.h5.zst
    # zstd -d /tmp/cube_single_expert.h5.zst -o ${STABLEWM_HOME}/ogbench/cube_single_expert.h5
    echo "  TODO: Set CUBE_DATASET_ID and uncomment gdown commands"
fi

if [ -d "${STABLEWM_HOME}/ogb_cube" ] && ls "${STABLEWM_HOME}/ogb_cube/"*ckpt* >/dev/null 2>&1; then
    echo "  Checkpoint: already present"
else
    mkdir -p "${STABLEWM_HOME}/ogb_cube"
    echo "  Checkpoint: downloading..."
    # TODO: Replace with actual Google Drive file ID
    # gdown <CUBE_CKPT_ID> -O ${STABLEWM_HOME}/ogb_cube/lejepa_object.ckpt
    echo "  TODO: Set CUBE_CKPT_ID and uncomment gdown commands"
fi

# -----------------------------------------------------------------------------
# DMControl Reacher
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] DMControl Reacher"
if [ -f "${STABLEWM_HOME}/dmc/reacher_random.h5" ]; then
    echo "  Dataset: already present"
else
    mkdir -p "${STABLEWM_HOME}/dmc"
    echo "  Dataset: downloading..."
    # TODO: Replace with actual Google Drive file ID
    # gdown <REACHER_DATASET_ID> -O /tmp/reacher_random.h5.zst
    # zstd -d /tmp/reacher_random.h5.zst -o ${STABLEWM_HOME}/dmc/reacher_random.h5
    echo "  TODO: Set REACHER_DATASET_ID and uncomment gdown commands"
fi

if [ -d "${STABLEWM_HOME}/dmc/reacher" ] && ls "${STABLEWM_HOME}/dmc/reacher/"*ckpt* >/dev/null 2>&1; then
    echo "  Checkpoint: already present"
else
    mkdir -p "${STABLEWM_HOME}/dmc/reacher"
    echo "  Checkpoint: downloading..."
    # TODO: Replace with actual Google Drive file ID
    # gdown <REACHER_CKPT_ID> -O ${STABLEWM_HOME}/dmc/reacher/lejepa_object.ckpt
    echo "  TODO: Set REACHER_CKPT_ID and uncomment gdown commands"
fi

# -----------------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "Volume contents:"
echo "=================================================="
ls -lh "${STABLEWM_HOME}/" 2>/dev/null || true
echo ""
echo "To find Google Drive IDs, check the stable-worldmodel repo or ask the model authors."
echo "Once IDs are set, re-run this script to download missing data."
