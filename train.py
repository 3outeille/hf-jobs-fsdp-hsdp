# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from AmineDiro/hfjobs-lora-buckets@29dc7d2 src/async_grpo_lora_buckets.py
"""Async GRPO + LoRA on Qwen3-30B-A3B across one or two HF Jobs, on transformers' own distributed API: their recipe
unchanged, plus LAYOUT=one|hsdp|fsdp for the mesh and a metrics.jsonl for the summarizer."""

import json
import logging
import os
import re
import threading
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback, set_seed
from transformers.distributed import DistributedConfig
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

MODEL_ID = "Qwen/Qwen3-30B-A3B"
LAYOUT = os.environ.get("LAYOUT", "one")
JOBS = int(os.environ.get("TRAINER_JOBS", "1"))
GPUS_PER_JOB = int(os.environ.get("GPUS_PER_JOB", "4"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", str(JOBS * GPUS_PER_JOB)))
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "16384"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "6"))          # the blog's value on 2 GPUs; scaled by WORLD_SIZE below
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "384"))
QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", "768"))
MAX_STALENESS = int(os.environ.get("MAX_STALENESS", "4"))
WEIGHT_SYNC_STEPS = int(os.environ.get("WEIGHT_SYNC_STEPS", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "20"))
MICROBATCH = int(os.environ.get("MICROBATCH", "1"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "3000"))
LORA_RANK = int(os.environ.get("LORA_RANK", "1"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/out")
RUN_NAME = os.environ.get("RUN_NAME", "moe")
SERVE_URL = os.environ.get("SERVE_URL", "http://localhost:8000")
METRICS = os.path.join(OUTPUT_DIR, "metrics.jsonl")

#: The attention projections: a MoE model's experts are fused 3-D parameters that PEFT cannot adapt once FSDP2 shards
#: them (moe-sft/docs/lora.md), and vLLM serves an adapter on q/k/v/o through its packed `qkv_proj` mapping.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

#: The rollout worker renders plain ChatML (their template, unchanged).
RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {"content": {"close_pattern": r"<\|im_end\|>\s*|$", "content": "text"}},
}

_SUBS = [(r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""), (r"\;", ""), (r"\ ", ""), (r"\$", ""),
         ("$", ""), (r"\%", ""), ("%", ""), (r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"), (r"^\circ", ""),
         (r"\circ", ""), (r"\text{", "{"), (r"\mbox{", "{")]

def _normalize(s):
    """Normalize LaTeX the way oat's grader does before comparing."""
    s = str(s).strip()
    for old, new in _SUBS:
        s = s.replace(old, new)
    s = re.sub(r"\\frac\s+", r"\\frac", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt\s+", r"\\sqrt", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".").removesuffix(".0")

def _last_boxed(text):
    """Contents of the LAST `\\boxed{...}` (oat uses rfind)."""
    start = text.rfind(r"\boxed")
    if start == -1:
        return None
    open_brace = text.find("{", start)
    if open_brace == -1:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    return None

def qwen_math_reward(completions, solution, completion_ids, **kwargs):
    """1.0 if the completion terminates and boxes the right answer, else 0.0 (their reward, unchanged)."""
    from math_verify import parse, verify

    on_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = 10 if on_main_thread else None
    verify_timeout = 5 if on_main_thread else None
    if not on_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)
    rewards = []
    for completion, gold, ids in zip(completions, solution, completion_ids, strict=True):
        if len(ids) >= MAX_COMPLETION_LENGTH:
            rewards.append(0.0)          # a rollout that hit the budget never collects a reward
            continue
        answer = _last_boxed(completion[0]["content"])
        if answer is None:
            rewards.append(0.0)
        elif _normalize(answer) == _normalize(gold):
            rewards.append(1.0)
        else:
            parsed_gold = parse(gold, parsing_timeout=parsing_timeout)
            parsed_answer = parse(answer, parsing_timeout=parsing_timeout)
            rewards.append(float(bool(parsed_gold) and bool(parsed_answer)
                                 and bool(verify(parsed_gold, parsed_answer, timeout_seconds=verify_timeout))))
    return rewards

def format_sample(sample):
    """The prompt is already conversational and ends with the boxed-answer instruction: only the gold has to move."""
    return {"prompt": sample["prompt"], "solution": sample["reward_model"]["ground_truth"]}

#: Resharding off for the base (as measured): one all-gather per micro-batch, not two.
FSDP_PLAN = {"model.layers.*": "keep_full_weight", "model.embed_tokens": "keep_full_weight"}

def enable_dtensor_shapes():
    """PEFT takes a sharded Linear's *local* shape, so the adapter is built for a fraction of the features and the forward
    dies with a size mismatch. FSDP2 shards storage, not compute: the full shape is the right one."""
    from peft.tuners import tuners_utils

    original = tuners_utils._get_in_out_features

    def patched(module):
        weight = getattr(module, "weight", None)
        mesh = getattr(weight, "device_mesh", None)
        # PEFT's local shape is right for tensor parallelism: only override a mesh with no "tp" dim (FSDP2/HSDP).
        if mesh is not None and "tp" not in (mesh.mesh_dim_names or ()):
            return weight.shape[1], weight.shape[0]      # (in, out) of the whole tensor
        return original(module)

    import sys   # PEFT's LoRA modules imported it by value: rebind it there too
    for module in list(sys.modules.values()):
        if getattr(module, "_get_in_out_features", None) is original:
            module._get_in_out_features = patched

def distributed_config():
    """LAYOUT -> the mesh transformers builds, and the accumulation that keeps a step's tokens (6 x 2 GPUs)."""
    if LAYOUT == "hsdp":                          # 2 Jobs x 2 GPUs: shard inside a machine, replicate across the two
        size, replicas = GPUS_PER_JOB, JOBS
    elif LAYOUT in ("fsdp", "one"):               # one mesh over every trainer GPU of the world
        size, replicas = JOBS * GPUS_PER_JOB, 1
    else:
        raise SystemExit(f"LAYOUT={LAYOUT}: use one, hsdp or fsdp")
    if TOKEN_BUDGET > 0 and (GRAD_ACCUM * 2) % WORLD_SIZE:
        raise SystemExit(f"GRAD_ACCUM={GRAD_ACCUM} x 2 does not divide WORLD_SIZE={WORLD_SIZE}")
    accum = GRAD_ACCUM * 2 // WORLD_SIZE if TOKEN_BUDGET > 0 else 128 // (MICROBATCH * WORLD_SIZE)
    return DistributedConfig(fsdp_size=size, dp_replicate_size=replicas, fsdp_plan=FSDP_PLAN), accum
def reset_adapter_init(trainer):
    """The adapter must start identical in every layout, and `from_pretrained` leaves the RNG in a mesh-dependent state (it
    draws per shard). So: one seed, then every LoRA layer re-initialised; Trainer broadcasts rank 0's values at train start."""
    from peft.tuners.lora.layer import LoraLayer
    set_seed(int(os.environ.get("SEED", "0")))
    for module in trainer.model.modules():
        if isinstance(module, LoraLayer):
            for adapter in module.lora_A:
                module.reset_lora_parameters(adapter, init_lora_weights=True)

class Recorder(TrainerCallback):
    """Every on_log dict, with its time and step, appended to $OUTPUT_DIR/metrics.jsonl on rank 0."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            os.makedirs(os.path.dirname(METRICS), exist_ok=True)
            with open(METRICS, "a") as out:
                out.write(json.dumps({"time": round(time.time(), 3), "step": state.global_step, **logs},
                                     default=float) + "\n")

def main():
    dc, accum = distributed_config()
    print(f"PARALLEL layout={LAYOUT} fsdp_size={dc.fsdp_size} dp_replicate_size={dc.dp_replicate_size} "
          f"plan=keep_full_weight world={WORLD_SIZE} grad_accum={accum} "
          f"gckpt={int(GRADIENT_CHECKPOINTING)} lora=attention:r{LORA_RANK}",
          flush=True)
    dataset = load_dataset("sail/Sanity-Test-R1D-1.5B", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.response_template = RESPONSE_TEMPLATE
    config = AsyncGRPOConfig(
        output_dir=OUTPUT_DIR, bf16=True, num_generations=8, temperature=1.0, top_p=1.0,
        max_completion_length=MAX_COMPLETION_LENGTH, learning_rate=4e-5, adam_beta2=0.95,
        per_device_train_batch_size=MICROBATCH, gradient_accumulation_steps=accum,
        token_budget=TOKEN_BUDGET, gradient_checkpointing=GRADIENT_CHECKPOINTING, max_staleness=MAX_STALENESS,
        max_inflight_tasks=MAX_INFLIGHT, queue_maxsize=QUEUE_MAXSIZE, weight_sync_steps=WEIGHT_SYNC_STEPS,
        max_steps=MAX_STEPS, save_strategy="steps", save_steps=SAVE_STEPS,
        ddp_timeout=7200,   # accelerate's PartialState makes the group; 2 h for a slow sharded load (default is 30 min)
        save_only_model=True,                  # this path refuses to resume, so optimizer state is not worth writing
        parallelism_config=None,
        model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True, "distributed_config": dc},
        vllm_server_base_url=SERVE_URL, run_name=RUN_NAME, report_to=[],
    )
    enable_dtensor_shapes()
    trainer = AsyncGRPOTrainer(
        model=MODEL_ID, args=config, train_dataset=dataset, processing_class=tokenizer,
        reward_funcs=qwen_math_reward,
        peft_config=LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, target_modules=LORA_TARGETS),
    )
    # transformers' own distributed loader: a sharded base, an unsharded (replicated) adapter.
    params = list(trainer.model.parameters())
    assert trainer.is_distributed_loading_by_transformers, "distributed_config did not reach from_pretrained"
    assert any(isinstance(p, torch.distributed.tensor.DTensor) for p in params), "the base is not sharded"
    assert not any(isinstance(p, torch.distributed.tensor.DTensor) for p in params if p.requires_grad), "adapter sharded"
    reset_adapter_init(trainer)
    trainer.add_callback(Recorder())
    trainer.train()


if __name__ == "__main__":
    main()
