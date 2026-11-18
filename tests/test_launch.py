"""The launcher's own decisions, against a stub API: what it ships, which machine it keeps, and what it cancels.

    uv run --with pytest --with huggingface_hub==2.4.0 python -m pytest tests/test_launch.py -q
"""
import contextlib, io, json, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch, summarize


class JobInfo:
    """Only what the Hub's Job object carries: the launcher must not lean on a `name` it does not have."""

    def __init__(self, id, boot):
        self.id, self.boot = id, boot


class FakeApi:
    """`run_job` takes the next boot id in `boots`; `cancel_job` records every attempt, `cancel_fails(n)` may raise."""

    def __init__(self, boots):
        self.boots, self.jobs, self.labels = list(boots), [], {}
        self.cancel_attempts, self.cancelled = [], []
        self.cancel_fails = lambda call: False

    def run_job(self, **spec):
        job = JobInfo(f"job{len(self.jobs)}", self.boots[len(self.jobs)])
        self.jobs.append(job)
        self.labels[job.id] = spec["name"]
        return job

    def fetch_job_logs(self, job_id, tail=50):
        return [f"BOOT {self.job(job_id).boot}"]

    def cancel_job(self, job_id):
        self.cancel_attempts.append(job_id)
        if self.cancel_fails(len(self.cancel_attempts)):
            raise RuntimeError(f"cancel {job_id} failed")
        self.cancelled.append(job_id)

    def job(self, job_id):
        return next(job for job in self.jobs if job.id == job_id)

    def names(self, ids):
        return [self.labels[job_id] for job_id in ids]

    def running(self):
        return self.names([job.id for job in self.jobs if job.id not in self.cancelled])


def test_upload_files_ships_only_the_code():
    root = Path(tempfile.mkdtemp())
    try:
        (root / "launch.py").write_text("code")
        for extra in (".git/HEAD", "results/hsdp/metrics.jsonl", "docs/hsdp-vs-fsdp.png"):
            (root / extra).parent.mkdir(parents=True, exist_ok=True)
            (root / extra).write_text("x")
        (root / "runs" / "20260101-000000" / "hsdp").mkdir(parents=True)
        (root / "runs" / "20260101-000000" / "hsdp" / "metrics.jsonl").write_text("{}\n")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "launch.cpython-313.pyc").write_bytes(b"")
        saved, launch.HERE = launch.HERE, root
        try:
            files = launch.upload_files("rid")
        finally:
            launch.HERE = saved
    finally:
        shutil.rmtree(root)
    assert files == [(str(root / "launch.py"), "code/rid/launch.py")], files


def test_place_returns_the_machine_it_kept_and_takes_its_blocker():
    api = FakeApi(["m1", "m1", "m2"])   # a Job on the machine in use, its blocker there, the retry elsewhere
    used = {"m1"}
    job, machine = launch.place(api, {"name": "train0", "image": "i", "flavor": "h200"}, used)
    assert (job.id, machine) == ("job2", "m2"), "the retry is kept, with the boot id read at its placement"
    assert used == {"m1", "m2"}, used
    assert api.names(api.cancelled) == ["train0", "blocker-train0-1"], "the rejected Job and its blocker, not the kept one"
    assert api.running() == ["train0"]


def test_place_retries_a_cancel_that_failed():
    api = FakeApi(["m1"])
    api.cancel_fails = lambda call: call == 1
    try:
        launch.place(api, {"name": "train0"}, used={"m1"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("a cancel that raised must not be swallowed")
    assert api.cancel_attempts == ["job0", "job0"], api.cancel_attempts
    assert api.running() == [], "the Job whose first cancel raised is cancelled again on the way out"


def test_place_reports_a_cancel_that_keeps_failing():
    api = FakeApi(["m1", "m1", "m1"])   # a rejected Job, its blocker on that machine, a retry that is rejected too
    api.cancel_fails = lambda call: call >= 2
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            launch.place(api, {"name": "train0", "image": "i", "flavor": "h200"}, used={"m1"}, attempts=2)
    except RuntimeError as error:
        assert "job2" in str(error), error
    else:
        raise AssertionError("a cancel that raised during the retry must not be swallowed")
    # the candidate cancelled inline, the retry's cancel that raised, then the finally's retries: the blocker and the retry
    assert api.cancel_attempts == ["job0", "job2", "job1", "job2"], api.cancel_attempts
    reported = errors.getvalue()
    assert "cancel job1" in reported and "cancel job2" in reported, f"every failed cancel is reported: {reported}"
    assert api.running() == ["blocker-train0-1", "train0"], "a cancel that fails leaves its Job running: said so, no more"


def test_place_that_gives_up_leaves_no_job_running():
    api = FakeApi(["m1"] * 6)           # three candidates and three blockers, every one on the machine in use
    try:
        launch.place(api, {"name": "train0", "image": "i", "flavor": "h200"}, used={"m1"}, attempts=3)
    except SystemExit:
        pass
    else:
        raise AssertionError("a placement that runs out of attempts must give up")
    assert api.running() == [], api.running()


def test_summarize_prints_throughput_without_a_unit():
    run = Path(tempfile.mkdtemp()) / "hsdp"
    run.mkdir(parents=True)
    rows = [{"step": 9 + i, "time": 10.0 * i, "perf/step_s": 2.0, "perf/fwd_bwd_s": 1.5, "perf/weight_sync_s": 0.5,
             "batch/trained_tokens_per_step": 1000} for i in range(3)]
    metrics = run / "metrics.jsonl"
    metrics.write_text("\n".join(json.dumps({"train/" + key: value for key, value in row.items()}) for row in rows))
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            summarize.main([str(metrics)])
    finally:
        shutil.rmtree(run.parent)
    row = [line for line in out.getvalue().splitlines() if line.startswith("| hsdp")][0]
    assert row == "| hsdp | 10.00 s | 2.00 s | 1.50 s | 0.50 s | 100 | 25 |", row
