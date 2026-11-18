#!/usr/bin/env bash
# The vLLM Job: the adapter sidecar on :8100, then vLLM on :8000 with LoRA enabled. No watchdog: launch.py cancels the
# replicas when the trainer Jobs end (and on Ctrl-C), and every Job has a --timeout as the backstop.
set -euo pipefail
echo "BOOT $(cat /proc/sys/kernel/random/boot_id)"   # launch.py reads this to warn if two Jobs share a machine
sleep "${PLACEMENT_GRACE_S:-0}"   # --isolate: a Job on a machine already in use is cancelled before it starts anything
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the code lives at /work/code/<run_id>/, not at /work
# Both as in the blog's run_vllm_job.sh: runtime LoRA loading, and the dev routes TRL calls (/server_info, /pause, /resume,
# /reset_prefix_cache), which vLLM 0.27.1 only serves in dev mode.
export VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 PYTHONUNBUFFERED=1
python3 "$HERE/sidecar.py" --port 8100 &
vllm serve Qwen/Qwen3-30B-A3B --host 0.0.0.0 --port 8000 --dtype bfloat16 \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --generation-config vllm \
    --logprobs-mode processed_logprobs --enable-lora --max-lora-rank 1 --max-loras 6 --max-cpu-loras 8 \
    --weight-transfer-config '{"backend": "nccl"}'
