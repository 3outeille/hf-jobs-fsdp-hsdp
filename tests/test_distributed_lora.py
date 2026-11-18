"""The adapter stays bit-identical across ranks on transformers' native FSDP2 path, and the two meshes agree.

    torchrun --nproc-per-node 4 --master-addr 127.0.0.1 --master-port <free> tests/test_distributed_lora.py --mode fsdp
    python3 tests/test_distributed_lora.py --mode baseline          # the plain single-process path
    torchrun ... --mode fsdp --no-sync                              # the negative control
"""
import argparse, hashlib, os, sys
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainerCallback
from transformers import TrainingArguments, set_seed
from transformers.distributed import DistributedConfig
TINY = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            moe_intermediate_size=16, intermediate_size=16, num_experts=4, num_experts_per_tok=2, vocab_size=128,
            max_position_embeddings=64, tie_word_embeddings=False)
PLAN = {"model.layers.*": "keep_full_weight", "model.embed_tokens": "keep_full_weight"}
def tiny_dir():
    path = os.environ["TINY_MODEL_DIR"]
    if not os.path.exists(os.path.join(path, "config.json")):
        torch.manual_seed(0)
        Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY)).save_pretrained(path)
    return path
def enable_dtensor_shapes():
    """PEFT reads a sharded `nn.Linear`'s shape from its DTensor and takes the *local* shard, so the adapter is built for a
    fraction of the features and the forward dies with a size mismatch. FSDP2 shards storage, not compute: use the whole
    shape. PEFT's LoRA modules imported the helper by value, so it is rebound there too."""
    from peft.tuners import tuners_utils
    original = tuners_utils._get_in_out_features

    def patched(module):
        weight = getattr(module, "weight", None)
        mesh = getattr(weight, "device_mesh", None)
        # PEFT's local shape is right for tensor parallelism: only override a mesh without a "tp" dimension (FSDP/HSDP).
        if mesh is not None and "tp" not in (mesh.mesh_dim_names or ()):
            return weight.shape[1], weight.shape[0]
        return original(module)

    tuners_utils._get_in_out_features = patched
    for module in list(sys.modules.values()):
        if getattr(module, "_get_in_out_features", None) is original:
            module._get_in_out_features = patched
def build(mode):
    """The base through the distributed loader, then TRL's order: LoRA after the sharding."""
    from peft import LoraConfig, get_peft_model
    config = {"fsdp": DistributedConfig(fsdp_size=4, fsdp_plan=PLAN),
              "hsdp": DistributedConfig(fsdp_size=2, dp_replicate_size=2, fsdp_plan=PLAN)}.get(mode)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir(), dtype="float32", distributed_config=config)
    enable_dtensor_shapes()
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                            task_type="CAUSAL_LM"))
def reset_adapter_init(trainer):
    """Production's re-init (train.py): one seed after the load, then every LoRA layer re-initialised."""
    from peft.tuners.lora.layer import LoraLayer
    set_seed(0)
    for module in trainer.model.modules():
        if isinstance(module, LoraLayer):
            for adapter in module.lora_A:
                module.reset_lora_parameters(adapter, init_lora_weights=True)
def digest(model):
    return hashlib.sha1(values(model, True).numpy().tobytes()).hexdigest()[:12]
def values(model, trainable):
    """The flattened parameters asked for, gathered to plain tensors (a DTensor's `full_tensor` is the whole value)."""
    out = []
    for _, param in model.named_parameters():
        if param.requires_grad == trainable:
            value = param.full_tensor() if hasattr(param, "full_tensor") else param.detach()
            out.append(value.detach().float().reshape(-1).cpu())
    return torch.cat(out)
def dataset():
    torch.manual_seed(1)
    return [{"input_ids": tokens, "labels": tokens.clone(), "attention_mask": torch.ones(12, dtype=torch.long)}
            for tokens in (torch.randint(0, TINY["vocab_size"], (12,)) for _ in range(8))]
def collate(rows):
    return {key: torch.stack([torch.as_tensor(row[key]) for row in rows])
            for key in ("input_ids", "labels", "attention_mask")}
class Checks(TrainerCallback):

    def __init__(self, output, base_before, adapter_before):
        self.output, self.base_before, self.adapter_before = output, base_before, adapter_before

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        adapter = values(model, True)
        torch.save({"adapter": adapter, "moved": not torch.equal(adapter, self.adapter_before),
                    "frozen": torch.equal(values(model, False), self.base_before)},
                   os.path.join(self.output, f"rank{os.environ.get('RANK', '0')}-step{state.global_step}.pt"))
def step_reports(output, steps=(1, 2, 3)):
    return {step: [torch.load(os.path.join(output, name), weights_only=False)
                   for name in sorted(os.listdir(output)) if name.endswith(f"-step{step}.pt")]
            for step in steps}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fsdp", "hsdp", "baseline"), required=True)
    parser.add_argument("--no-sync", action="store_true", help="negative control: no adapter all-reduce")
    args = parser.parse_args()
    distributed = args.mode != "baseline"
    if distributed and not dist.is_initialized():
        dist.init_process_group("gloo")
    if args.no_sync:
        Trainer._sync_replicated_trainable_parameters = lambda self, *a, **k: None
    model = build(args.mode)
    base_before = values(model, False)
    output = os.path.join(os.environ.get("RUN_DIR", "/tmp/run"), args.mode + ("-nosync" if args.no_sync else ""))
    checks = Checks(output, base_before, None)   # its adapter_before is set after the re-init
    trainer = Trainer(model=model, train_dataset=dataset(), data_collator=collate,
                      args=TrainingArguments(output_dir=output, max_steps=3, per_device_train_batch_size=1,
                                             gradient_accumulation_steps=2, save_strategy="steps", save_steps=3,
                                             save_only_model=True, use_cpu=True, report_to=[], learning_rate=1e-2,
                                             max_grad_norm=1.0, seed=0, logging_steps=1),
                      callbacks=[checks])
    reset_adapter_init(trainer)     # the seed is *after* the load, as in production
    print(f"ADAPTER INIT rank={os.environ.get('RANK', '0')} mode={args.mode} sha1="
          f"{hashlib.sha1(values(trainer.model, True).numpy().tobytes()).hexdigest()[:12]}", flush=True)
    checks.adapter_before = values(trainer.model, True)
    first = next(iter(trainer.get_train_dataloader()))["input_ids"]
    print(f"BATCH rank={os.environ.get('RANK', '0')} sha1="
          f"{hashlib.sha1(first.numpy().tobytes()).hexdigest()[:12]}", flush=True)
    trainer.train()
    rank, adapter = int(os.environ.get("RANK", "0")), values(trainer.model, True)
    reports = step_reports(output)
    same = {step: all(torch.equal(items[0]["adapter"], other["adapter"]) for other in items[1:])
            for step, items in reports.items()}
    if args.no_sync and distributed:
        assert reports and not any(same.values()), f"the negative control did not diverge ({same})"
        if rank == 0:
            print("NEGATIVE CONTROL OK: the ranks diverge without the adapter all-reduce", flush=True)
        return 0
    assert reports and all(same.values()), f"the adapter differed across ranks: {same}"
    assert all(item["frozen"] for items in reports.values() for item in items), "the frozen base moved"
    assert all(item["moved"] for items in reports.values() for item in items), "the adapter never changed"
    assert os.path.exists(os.path.join(output, "checkpoint-3", "adapter_config.json")), "no checkpoint with save_only_model"
    torch.save(adapter, os.path.join(os.environ["TINY_MODEL_DIR"], f"adapter-{args.mode}.pt"))
    other = os.path.join(os.environ["TINY_MODEL_DIR"], f"adapter-{'hsdp' if args.mode == 'fsdp' else 'fsdp'}.pt")
    if rank == 0 and args.mode != "baseline" and os.path.exists(other):
        # With the seeded adapter init (see build()) the two meshes are bit-identical here: without it PEFT's LoRA A/B is
        # drawn from RNG states the distributed load leaves different per mesh, and the weights drift by ~0.34 in 3 steps.
        difference = (adapter - torch.load(other)).abs().max().item()
        assert difference <= 1e-4, f"{args.mode} and the other mesh differ by {difference}"
        print(f"MESH MATCH OK: {args.mode} == the other mesh (max|diff|={difference:.1e})", flush=True)
    return 0
if __name__ == "__main__":
    sys.exit(main())
