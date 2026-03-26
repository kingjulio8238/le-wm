#!/bin/bash
# =============================================================================
# Phase D1: Multi-Task Validation
#
# Runs eval.py on all 4 tasks and collects results.
# Prerequisites: datasets + checkpoints downloaded to $STABLEWM_HOME
#
# Usage:
#   export STABLEWM_HOME=/workspace/data
#   cd /workspace/le-harness
#   bash scripts/run_d1_multitask.sh
# =============================================================================

set -euo pipefail

STABLEWM_HOME="${STABLEWM_HOME:-/workspace/data}"
RESULTS_DIR="${STABLEWM_HOME}/results/d1"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"

export STABLEWM_HOME
export MUJOCO_GL=egl

# Task definitions: config_name|policy_path|display_name
TASKS=(
    "pusht|pusht/lejepa|PushT"
    "tworoom|tworoom/lejepa|TwoRoom"
    "cube|ogb_cube/lejepa|Cube"
    "reacher|dmc/reacher/lejepa|Reacher"
)

SUMMARY_FILE="${RESULTS_DIR}/d1_summary.txt"
echo "Phase D1: Multi-Task Validation Results" > "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "Date: $(date -Iseconds)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for task_spec in "${TASKS[@]}"; do
    IFS='|' read -r config_name policy_path display_name <<< "$task_spec"

    echo ""
    echo "================================================================"
    echo "  Evaluating: ${display_name} (config=${config_name}, policy=${policy_path})"
    echo "================================================================"

    log_file="${LOG_DIR}/d1_${config_name}.log"

    # Check if checkpoint exists
    ckpt_dir="${STABLEWM_HOME}/${policy_path}"
    if [ ! -d "$ckpt_dir" ] && [ ! -f "${ckpt_dir}.ckpt" ] && [ ! -f "${ckpt_dir}_object.ckpt" ]; then
        echo "  SKIP: Checkpoint not found at ${ckpt_dir}"
        echo "${display_name}: SKIPPED (checkpoint not found at ${ckpt_dir})" >> "$SUMMARY_FILE"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # Run eval
    start_time=$(date +%s)
    if python eval.py --config-name="$config_name" policy="$policy_path" 2>&1 | tee "$log_file"; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "  DONE in ${elapsed}s. Log: ${log_file}"

        # Extract metrics from log (eval.py prints metrics dict)
        metrics_line=$(grep -E "^(\{|metrics:)" "$log_file" | tail -1 || echo "")
        echo "${display_name}: SUCCESS (${elapsed}s)" >> "$SUMMARY_FILE"
        echo "  Log: ${log_file}" >> "$SUMMARY_FILE"
        if [ -n "$metrics_line" ]; then
            echo "  Metrics: ${metrics_line}" >> "$SUMMARY_FILE"
        fi
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "  FAILED after ${elapsed}s. See: ${log_file}"
        echo "${display_name}: FAILED (${elapsed}s)" >> "$SUMMARY_FILE"
        echo "  Log: ${log_file}" >> "$SUMMARY_FILE"

        # Capture last 10 lines of error
        echo "  Last 10 lines:" >> "$SUMMARY_FILE"
        tail -10 "$log_file" | sed 's/^/    /' >> "$SUMMARY_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo "" >> "$SUMMARY_FILE"
done

# Summary
echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "TOTALS: ${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped" >> "$SUMMARY_FILE"
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo "Gate: >50% success on >=3/4 tasks needed" >> "$SUMMARY_FILE"

echo ""
echo "================================================================"
echo "  D1 SUMMARY"
echo "================================================================"
echo "  Passed:  ${PASS_COUNT}"
echo "  Failed:  ${FAIL_COUNT}"
echo "  Skipped: ${SKIP_COUNT}"
echo ""
echo "  Full summary: ${SUMMARY_FILE}"
echo "  Logs: ${LOG_DIR}/"
echo "================================================================"

cat "$SUMMARY_FILE"
