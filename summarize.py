"""The three rows of the MoE comparison, from the runs' own metrics.jsonl.

    python3 summarize.py runs/<run id>/metrics.jsonl ...

Steps 9-30, and the same definitions as the earlier equal-GPU table: wall time per step = the span between the
first and the last of them over n-1 intervals, trained tokens/s = the tokens trained in those steps over that span (so
the steps that wait for a weight sync count), medians over every measured step for the per-step numbers.
"""
import json, statistics, sys
from pathlib import Path

FIRST = 9


def rows(path):
    """The measured steps. Keys may arrive with TRL's `train/` prefix or bare (our callback writes the log dict as is)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = {(key[len("train/"):] if key.startswith("train/") else key): value
                      for key, value in json.loads(line).items()}
            if isinstance(record.get("perf/step_s"), (int, float)):
                out.append(record)
    return [row for row in out if row.get("step", 0) >= FIRST]


def measure(path):
    measured = rows(path)
    timed = sorted(measured, key=lambda row: row["time"])
    span = timed[-1]["time"] - timed[0]["time"] if len(timed) > 1 else None
    tokens = sum(row.get("batch/trained_tokens_per_step", 0) for row in timed[1:])
    step = [row["perf/step_s"] for row in timed]
    fwd = [row["perf/fwd_bwd_s"] for row in timed if isinstance(row.get("perf/fwd_bwd_s"), (int, float))]
    sync = [row["perf/weight_sync_s"] for row in timed if isinstance(row.get("perf/weight_sync_s"), (int, float))]
    meta = Path(path).parent / "run.json"
    gpus = int(json.loads(meta.read_text()).get("trainer_gpus", 4)) if meta.exists() else 4
    return {"name": Path(path).parent.name, "wall": span / (len(timed) - 1) if span else None,
            "tokens": tokens / span if span else None, "per_gpu": tokens / span / gpus if span else None,
            "step": statistics.median(step) if step else None, "fwd": statistics.median(fwd) if fwd else None,
            "sync": statistics.median(sync) if sync else None}


def cell(value, unit="s", digits=0):
    return "-" if value is None else f"{value:,.{digits}f}{unit}"


def main(paths):
    print("| run | wall s/step | step p50 | fwd+bwd p50 | sync p50 | trained tok/s | tok/s per GPU |")
    print("|---|---|---|---|---|---|---|")
    for row in (measure(path) for path in paths):
        print(f"| {row['name']} | {cell(row['wall'], ' s', 2)} | {cell(row['step'], ' s', 2)} "
              f"| {cell(row['fwd'], ' s', 2)} | {cell(row['sync'], ' s', 2)} | {cell(row['tokens'], '')} "
              f"| {cell(row['per_gpu'], '')} |")


if __name__ == "__main__":
    main(sys.argv[1:])
