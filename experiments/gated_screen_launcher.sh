#!/bin/bash
# Waits for the HF token, then runs the base screen on the three gated models.
# Runs inside tmux so it survives the agent harness; all output goes to the log.
cd /workspace/relocation
LOG=results/screen_base_gated.log
echo "$(date -u +%FT%TZ) waiting for /workspace/hf/token" | tee -a "$LOG"
until [ -f /workspace/hf/token ]; do sleep 5; done
echo "$(date -u +%FT%TZ) TOKEN LANDED" | tee -a "$LOG"
used_g=$(du -s /workspace 2>/dev/null | awk '{printf "%d", $1/1048576}')
if [ "$used_g" -gt 36 ]; then
  echo "DISK GUARD: /workspace at ${used_g}G, not launching. GATED_SCREEN_ABORTED" | tee -a "$LOG"
  exit 1
fi
echo "launching at ${used_g}G used" | tee -a "$LOG"
source venv/bin/activate
export HF_HOME=/workspace/hf
python -u experiments/screen_base.py google/gemma-2-2b meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B 2>&1 | tee -a "$LOG"
echo "GATED_SCREEN_EXITED rc=${PIPESTATUS[0]}" | tee -a "$LOG"
