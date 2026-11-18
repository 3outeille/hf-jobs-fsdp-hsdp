"""Run the Qwen3-30B-A3B comparison on HF Jobs: one vLLM Job per replica, then one or two trainer Jobs.

    python3 launch.py --layout hsdp            # 2 trainer Jobs of h200x2 (HSDP 2x2), 3 replicas, 30 steps
    python3 launch.py --layout fsdp            # the same four GPUs, one FSDP mesh over them
    python3 launch.py --layout one             # a single h200x4 Job (FSDP over its own GPUs)
    python3 launch.py --layout hsdp --dry-run  # print the Jobs and exit

The result is the comparison: `one` shows what a single Job buys, `hsdp` keeps it across two Jobs, and `fsdp` pays for the
frozen base over the network on every micro-batch. Run it once per layout; `summarize.py` puts the rows side by side.

Only `huggingface_hub`: the folder goes to a Storage Bucket under `code/<run_id>/` and every Job mounts that bucket
(read-only at /work; the trainer also mounts it read-write at /out). No preflight, no orchestrator, no reaper, no series
mode: the replicas are cancelled when the trainer Jobs end (or on Ctrl-C), and each Job has a --timeout as the backstop.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi, Volume, get_token
except ImportError:   # a --dry-run on a machine without the pin: the plan still prints
    HfApi = Volume = get_token = None

HERE = Path(__file__).resolve().parent
IMAGE = "vllm/vllm-openai:v0.27.1"
TRL_SPEC = "https://codeload.github.com/huggingface/trl/tar.gz/9d2f38056aa0c79be500525536b2830fc47dd35a"
#: transformers main at 75ca973: its native FSDP2 path needs torch>=2.8 (the image's 2.13 does) and it is what reads
#: `distributed_config` and the `fsdp_plan` overrides.
TRANSFORMERS_SPEC = "https://codeload.github.com/huggingface/transformers/tar.gz/75ca973171ee654e288efc16dd9faf4f3051c014"
#: kernels==0.18.1: transformers @75ca973 requires kernels>=0.18.0,<0.19 (it raises `_MISSING_KERNELS_MESSAGE` when TRL asks
#: for `attn_implementation="kernels-community/flash-attn3"`); the image's 0.16.1 is too old, and the trainer Job installs on
#: top of it. 0.18.1 is the last 0.18.x before that commit (2026-11-02) and wants huggingface-hub>=1.32.0 (2.4.0 is fine).
EXTRA_SPECS = "peft==0.20.0 kernels==0.18.1 math-verify==0.9.0 latex2sympy2_extended==1.11.0"
#: layout -> (trainer Jobs, GPUs per Job). `one` is a single h200x4; the others are two Jobs of two GPUs.
LAYOUTS = {"one": (1, 4), "hsdp": (2, 2), "fsdp": (2, 2)}
#: The knobs the README lists, passed through from the environment (the defaults are the measured runs' values).
#: They reach the trainer Jobs only, never the vLLM replicas.
KNOBS = {"MAX_STEPS": "30", "TOKEN_BUDGET": "16384", "GRAD_ACCUM": "6", "GRADIENT_CHECKPOINTING": "1",
         "MAX_INFLIGHT": "384", "QUEUE_MAXSIZE": "768", "MAX_STALENESS": "4", "WEIGHT_SYNC_STEPS": "4",
         "SAVE_STEPS": "20", "LORA_RANK": "1", "MAX_COMPLETION_LENGTH": "3000"}
ENDED = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def volume(bucket, mount_path, read_only):
    """A bucket as a volume; a plain dict when huggingface_hub is absent (a dry run prints, never launches)."""
    if Volume is None:
        return {"type": "bucket", "source": bucket, "mount_path": mount_path, "read_only": read_only}
    return Volume(type="bucket", source=bucket, mount_path=mount_path, read_only=read_only)


def plan(args, run_id, jobs, gpus, token):
    """The Jobs of one layout, as run_job kwargs. The trainers' knobs come from the environment."""
    code, out = f"code/{run_id}", f"runs/{run_id}/{args.layout}"
    knobs = {key: os.environ.get(key, default) for key, default in KNOBS.items()}
    knobs["MAX_STEPS"] = str(args.steps)
    trainer_env = dict(knobs, LAYOUT=args.layout, TRAINER_JOBS=str(jobs), GPUS_PER_JOB=str(gpus),
                       OUTPUT_DIR=f"/out/{out}", SERVE_URL="http://localhost:8000", TRL_SPEC=TRL_SPEC,
                       TRANSFORMERS_SPEC=TRANSFORMERS_SPEC, EXTRA_SPECS=EXTRA_SPECS, REPLICAS=str(args.replicas),
                       RUN_NAME=f"{args.layout}-{run_id}")
    if jobs > 1:   # NCCL's one TCP socket per connection uses a quarter of the link between Jobs; 8 x 8 is what was measured
        trainer_env.update(NCCL_SOCKET_NTHREADS="8", NCCL_NSOCKS_PERTHREAD="8")
    common = dict(image=IMAGE, timeout=args.timeout, network_group=args.group, secrets={"HF_TOKEN": token})
    replicas = [dict(common, command=["bash", f"/work/{code}/serve.sh"], flavor="h200",
                     name=f"agrpo-{run_id}-moe-vllm{i}", network_aliases=[f"vllm{i}"], env={},
                     volumes=[volume(args.bucket, "/work", True)]) for i in range(1, args.replicas + 1)]
    trainers = [dict(common, command=["bash", f"/work/{code}/train.sh"], flavor=f"h200x{gpus}",
                     name=f"agrpo-{run_id}-moe-train{rank}", network_aliases=[f"trainer{rank}"],
                     env=dict(trainer_env, MACHINE_RANK=str(rank)),
                     volumes=[volume(args.bucket, "/work", True), volume(args.bucket, "/out", False)])
                for rank in range(jobs)]
    return replicas, trainers


def boot_id(api, job_id, tries=20):
    """The machine a Job landed on: every Job body prints its `/proc/sys/kernel/random/boot_id` first."""
    for _ in range(tries):
        try:
            for line in api.fetch_job_logs(job_id=job_id, tail=50):   # an Iterable[str], not one string
                if line.startswith("BOOT "):
                    return line.split()[1]
        except Exception:  # noqa: BLE001 - a Job whose log is not up yet is not an error
            pass
        time.sleep(6)
    return None


def place(api, spec, used, attempts=10):
    """`--isolate`: start one Job and keep it only on a machine no other Job of this run uses; return it with that
    machine (the boot id it was decided on, so the report never reads the log again). A rejected Job's slot is taken by
    a blocker (same flavor and image, so the scheduler, which prefers that machine, fills it) until the retry lands
    elsewhere; the blockers go as soon as the Job is placed. Every Job this call starts and does not return is cancelled
    on the way out, each cancel reported if it fails: a placement that gives up tries to leave nothing running."""
    started = []                                   # Jobs this call started and still owns
    try:
        for attempt in range(1, attempts + 1):
            job = api.run_job(**spec)
            started.append(job)
            machine = boot_id(api, job.id)
            if machine and machine not in used:
                used.add(machine)
                started.remove(job)
                return job, machine
            api.cancel_job(job_id=job.id)
            started.remove(job)
            print(f"{spec['name']} landed on {machine}, already used by this run: cancelled it", flush=True)
            blocker = api.run_job(image=spec["image"], flavor=spec["flavor"], timeout="30m",
                                  name=f"blocker-{spec['name']}-{attempt}",
                                  command=["bash", "-c", "echo BOOT $(cat /proc/sys/kernel/random/boot_id); sleep 3600"])
            started.append(blocker)
            if boot_id(api, blocker.id) not in used:
                api.cancel_job(job_id=blocker.id)
                started.remove(blocker)
        raise SystemExit(f"no machine of its own for {spec['name']} after {attempts} tries")
    finally:
        for job in started:                        # a cancel that raised left its Job running: retried here
            try:
                api.cancel_job(job_id=job.id)
            except Exception as error:  # noqa: BLE001 - keep cancelling the rest
                print(f"cancel {job.id}: {error}", file=sys.stderr)


def stage_of(api, job_id):
    """The stage as a plain string: `status.stage` is an enum whose `str()` is not the value."""
    status = api.inspect_job(job_id=job_id).status
    stage = getattr(status, "stage", status)
    return getattr(stage, "value", str(stage))


def upload_files(run_id):
    """The folder as bucket files under `code/<run_id>/`: only the code, not `.git`, `runs/`, `results/`, `docs/` or `__pycache__`."""
    return [(str(path), f"code/{run_id}/{path.relative_to(HERE).as_posix()}")
            for path in sorted(HERE.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.relative_to(HERE).parts[0] not in ("runs", "results", "docs")
            and not path.relative_to(HERE).parts[0].startswith(".")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="hsdp")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--bucket", default=None, help="namespace/name; created if missing")
    parser.add_argument("--timeout", default="1h", help="per Job, the backstop for a run that lost its launcher")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isolate", action="store_true",
                        help="every Job on a machine of its own (blocker Jobs and retries: slower, costs more)")
    args = parser.parse_args()
    api = None if args.dry_run else HfApi()
    # `get_token()` reads HF_TOKEN or the `hf auth login` store: `api.token` is None for the latter.
    token = os.environ.get("HF_TOKEN") or (get_token() if api else None)
    args.bucket = args.bucket or (f"{api.whoami()['name']}/hf-jobs-fsdp-hsdp" if api else "me/hf-jobs-fsdp-hsdp")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    args.group = f"agrpo-moe-{run_id}"
    jobs, gpus = LAYOUTS[args.layout]
    replicas, trainers = plan(args, run_id, jobs, gpus, token)
    if args.isolate:   # each Job waits before it starts work, so a rejected one is cancelled before it computed anything
        for spec in replicas + trainers:
            spec["env"] = dict(spec["env"], PLACEMENT_GRACE_S="60")
    print(f"# {args.layout}: {jobs} trainer Job(s) of h200x{gpus} + {args.replicas} h200 replicas, {args.steps} steps",
          flush=True)
    for spec in replicas + trainers:
        print(f"#   {spec['name']}: {spec['flavor']} {spec['command'][-1]} aliases={spec['network_aliases']}",
              flush=True)
    if args.dry_run:
        for spec in trainers:
            print(json.dumps({key: spec[key] for key in ("name", "flavor", "network_aliases", "volumes", "env")},
                             default=str)[:500])
        return 0
    if not token:
        raise SystemExit("no token: export HF_TOKEN (every Job gets it as a secret)")
    api.create_bucket(args.bucket, private=True, exist_ok=True)
    files = upload_files(run_id)
    api.batch_bucket_files(args.bucket, add=files)
    print(f"# shipped {len(files)} file(s) to {args.bucket}/code/{run_id}", flush=True)
    machine_of, launched = {}, []                      # launched: (name, job) of every Job started, trainers included
    try:
        used, placement = set(), {}                    # placement: name -> the boot id read when the Job was placed
        for spec in replicas + trainers:
            if args.isolate:
                job, placement[spec["name"]] = place(api, spec, used)   # place() decided on this boot id
            else:
                job = api.run_job(**spec)              # every Job starts first: no boot id is read before the last one
            launched.append((spec["name"], job))
        trainer_jobs = launched[len(replicas):]
        for name, job in launched:                     # one read per Job for the other layouts, right after the launches
            machine = placement[name] if args.isolate else boot_id(api, job.id)
            machine_of.setdefault(machine, []).append(name)
            print(f"{name} {job.id} machine={machine}", flush=True)
        # Two Jobs of this run on one machine make the row's numbers meaningless (a trainer pair especially).
        for machine, names in machine_of.items():
            if machine is not None and len(names) > 1:
                print(f"WARNING: {len(names)} Jobs share machine {machine}: {', '.join(names)}", file=sys.stderr)
        while True:
            stages = {name: stage_of(api, job.id) for name, job in trainer_jobs}
            print(f"{time.strftime('%H:%M:%S')} {stages}", flush=True)
            if any(stage in ("ERROR", "CANCELED") for stage in stages.values()):
                print("a trainer Job failed or was cancelled: stopping the run", file=sys.stderr)
                break
            if all(stage in ENDED for stage in stages.values()):
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("interrupted: cancelling every Job of this run", file=sys.stderr)
    finally:
        for name, job in launched:
            try:
                if stage_of(api, job.id) not in ENDED:
                    api.cancel_job(job_id=job.id)
            except Exception as error:  # noqa: BLE001 - say so, keep cancelling the rest
                print(f"cancel {name}: {error}", file=sys.stderr)
    local = HERE / "runs" / run_id / args.layout
    local.mkdir(parents=True, exist_ok=True)
    try:
        api.download_bucket_files(args.bucket, files=[(f"runs/{run_id}/{args.layout}/metrics.jsonl",
                                                       str(local / "metrics.jsonl"))])
    except Exception as error:  # noqa: BLE001 - report it, do not hide the trainer's failure
        print(f"could not fetch metrics.jsonl: {error}", file=sys.stderr)
    (local / "run.json").write_text(json.dumps({"layout": args.layout, "trainer_gpus": jobs * gpus,
                                                "replicas": args.replicas, "steps": args.steps, "run_id": run_id}))
    if (local / "metrics.jsonl").exists():
        subprocess.run([sys.executable, str(HERE / "summarize.py"), str(local / "metrics.jsonl")], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
