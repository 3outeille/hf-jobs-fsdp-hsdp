#!/usr/bin/env bash
# The trainer Job body. Rank 0 of the first Job runs the gateway on :8000 (TRL's SERVE_URL); every rank then runs the same
# torchrun, which sets RANK/LOCAL_RANK/WORLD_SIZE for transformers' own distributed API (distributed_config on the model).
set -euo pipefail
echo "BOOT $(cat /proc/sys/kernel/random/boot_id)"   # launch.py reads this to warn if two Jobs share a machine
sleep "${PLACEMENT_GRACE_S:-0}"   # --isolate: a Job on a machine already in use is cancelled before it starts anything
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the code lives at /work/code/<run_id>/, not at /work
# Only Python-only packages: torch and vllm stay the image's (the trainer imports vllm for TRL's weight-transfer client).
pip install -q "huggingface_hub==2.4.0" "${TRANSFORMERS_SPEC}" "${TRL_SPEC}" ${EXTRA_SPECS} ${TRACKIO_SPEC:-}
python3 - <<'PY' || { echo "the trainer's stack is broken after the install: not starting"; exit 1; }
import accelerate, kernels, peft, torch, tokenizers, transformers, vllm
from transformers.utils import is_kernels_available
print(f"torch {torch.__version__} transformers {transformers.__version__} tokenizers {tokenizers.__version__} "
      f"kernels {kernels.__version__} vllm {vllm.__version__} peft {peft.__version__} "
      f"accelerate {accelerate.__version__} kernels_available={is_kernels_available()}")
# TRL asks for attn_implementation="kernels-community/flash-attn3"; transformers raises before the first step when its
# kernels pin (>=0.18,<0.19 at 75ca973) is not met by what the image ships plus what this line installs.
assert is_kernels_available(), "is_kernels_available() is false: the FLASH-ATTN3 kernel would raise at from_pretrained"
# kernels checks the attention kernel's sigstore signature, and the first check in a container writes sigstore's trust
# metadata to disk: the ranks of one Job doing it at once race on the same files (FileExistsError on root.json). Once here.
from sigstore.verify.verifier import Verifier
Verifier.production()
PY
cd "$HERE"
if [[ "${MACHINE_RANK}" == 0 ]]; then
    python3 "$HERE/gateway.py" &
    for _ in $(seq 60); do curl -sf http://localhost:8000/health > /dev/null && break; sleep 2; done
fi
torchrun --nnodes "${TRAINER_JOBS}" --nproc-per-node "${GPUS_PER_JOB}" --node-rank "${MACHINE_RANK}" \
    --rdzv-backend static --master-addr "${HF_NETWORK_GROUP_PREFIX}trainer0" --master-port 29500 "$HERE/train.py"
