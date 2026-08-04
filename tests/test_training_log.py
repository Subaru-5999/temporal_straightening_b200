"""Tests for training_log.py and summarize_training_log.py.

The load-bearing claim is that memory is bounded: O(#metrics), not O(#steps).
That is asserted directly by feeding 200k steps and checking that nothing in the
logger grew. The rest covers Welford correctness, anomaly detection, and that a
truncated log (run killed mid-write) still summarises.

Run:  pytest tests/test_training_log.py -q
"""

import json
import math
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training_log import OnlineStats, TrainingLogger


# ------------------------------------------------------------------- OnlineStats
def test_welford_matches_numpy_style_reference():
    xs = [1.5, -2.0, 3.25, 0.0, 7.5, -1.25, 4.0]
    st = OnlineStats()
    for x in xs:
        st.update(x)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    assert st.n == n
    assert st.mean == pytest.approx(mean)
    assert st.var == pytest.approx(var)
    assert st.std == pytest.approx(math.sqrt(var))
    assert st.min == min(xs) and st.max == max(xs) and st.last == xs[-1]


def test_welford_is_numerically_stable_on_a_large_offset():
    """The naive sum-of-squares formula loses all precision here; Welford does not."""
    st = OnlineStats()
    for x in (1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16):
        st.update(x)
    assert st.std == pytest.approx(5.4772255, rel=1e-4)


def test_nonfinite_is_counted_not_averaged():
    st = OnlineStats()
    st.update(1.0)
    st.update(float("nan"))
    st.update(float("inf"))
    st.update(3.0)
    assert st.n == 2
    assert st.mean == pytest.approx(2.0)
    assert st.n_nonfinite == 2
    assert st.as_dict()["nonfinite"] == 2


def test_empty_stats_serialise():
    assert OnlineStats().as_dict() == {"n": 0, "nonfinite": 0}


def test_online_stats_is_fixed_size():
    """__slots__, so no per-sample attribute growth."""
    st = OnlineStats()
    assert not hasattr(st, "__dict__")
    for i in range(10_000):
        st.update(i)
    assert st.n == 10_000


# ---------------------------------------------------------------- memory bounds
def test_memory_does_not_grow_with_steps(tmp_path):
    """The central claim. 200k steps must leave the logger the same size."""
    tl = TrainingLogger(tmp_path / "t.jsonl", run_name="mem", log_every=1000)
    for step in range(1, 2001):
        tl.record(step, **{"loss/total": 1.0 / step, "loss/sigreg": 2.0})
        tl.maybe_flush(step)
    early = tl.memory_report()

    for step in range(2001, 200_001):
        tl.record(step, **{"loss/total": 1.0 / step, "loss/sigreg": 2.0})
        tl.maybe_flush(step)
    late = tl.memory_report()

    assert late["metric_keys_lifetime"] == early["metric_keys_lifetime"]
    assert late["events_retained"] <= late["events_cap"]
    assert late["intervals_retained"] <= late["intervals_cap"]
    assert late["grows_with_steps"] is False
    # 100x more steps, same number of tracked keys
    assert late["metric_keys_lifetime"] == 2


def test_event_ring_is_capped(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl", ring_events=10)
    for i in range(500):
        tl.event(i, "synthetic", f"event {i}")
    assert len(tl.events) == 10
    assert tl.events[-1]["msg"] == "event 499"        # newest retained
    assert tl._n_events == 500                        # count is still exact
    # but every one of them reached disk
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert sum(1 for l in lines if json.loads(l).get("type") == "event") == 500


def test_interval_accumulators_reset_on_flush(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl", log_every=10)
    for step in range(1, 21):
        tl.record(step, **{"loss/total": float(step)})
        tl.maybe_flush(step)
    recs = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().strip().split("\n")]
    intervals = [r for r in recs if r["type"] == "interval"]
    assert len(intervals) == 2
    # windows must be disjoint: 1..10 then 11..20
    assert intervals[0]["metrics"]["loss/total"]["mean"] == pytest.approx(5.5)
    assert intervals[1]["metrics"]["loss/total"]["mean"] == pytest.approx(15.5)


# ------------------------------------------------------------ anomaly detection
def test_nonfinite_loss_raises_an_event(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl")
    tl.record(7, **{"loss/total": float("nan")})
    kinds = [e["kind"] for e in tl.events]
    assert "nonfinite" in kinds
    assert tl.events[-1]["step"] == 7


def test_loss_spike_is_detected_against_the_running_mean(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl", spike_sigma=5.0, spike_min_n=50)
    for step in range(300):
        tl.record(step, **{"loss/total": 1.0 + 0.001 * (step % 7)})
    assert not [e for e in tl.events if e["kind"] == "loss_spike"]
    tl.record(999, **{"loss/total": 50.0})
    spikes = [e for e in tl.events if e["kind"] == "loss_spike"]
    assert len(spikes) == 1 and spikes[0]["step"] == 999


def test_no_spike_before_enough_history(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl", spike_sigma=3.0, spike_min_n=100)
    for step in range(20):
        tl.record(step, **{"loss/total": 1.0})
    tl.record(21, **{"loss/total": 99.0})
    assert not [e for e in tl.events if e["kind"] == "loss_spike"]


def test_collapse_thresholds_fire(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl")
    tl.probe_latents(500, {"latent/probe_r2": 0.8, "latent/eff_rank_frac": 0.5,
                           "latent/std": 0.3})
    assert not [e for e in tl.events if e["kind"] == "collapse_warning"]
    tl.probe_latents(1000, {"latent/probe_r2": 0.0001, "latent/eff_rank_frac": 0.05,
                            "latent/std": 1e-6})
    warns = [e for e in tl.events if e["kind"] == "collapse_warning"]
    assert len(warns) == 3
    assert all(w["step"] == 1000 for w in warns)


def test_nan_diagnostics_do_not_fire_collapse(tmp_path):
    """NaN means 'not computable', not 'collapsed'."""
    tl = TrainingLogger(tmp_path / "t.jsonl")
    tl.probe_latents(10, {"latent/probe_r2": float("nan")})
    assert not [e for e in tl.events if e["kind"] == "collapse_warning"]


# --------------------------------------------------------------------- robustness
def test_logger_never_raises_on_bad_input(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl")
    tl.record(1, a=None, b="not a number", c=[1, 2], d=3.0)
    tl.record_dict(1, None)
    tl.probe_latents(1, None)
    tl.flush(1)
    tl.close(1)
    tl.close(1)                                   # idempotent
    assert tl._closed


def test_disabled_logger_writes_nothing(tmp_path):
    p = tmp_path / "t.jsonl"
    tl = TrainingLogger(p, enabled=False)
    tl.record(1, **{"loss/total": 1.0})
    tl.event(1, "x", "y")
    tl.flush(1)
    tl.close(1)
    assert not p.exists()


def test_write_failure_disables_telemetry_instead_of_crashing(tmp_path):
    tl = TrainingLogger(tmp_path / "t.jsonl")
    tl.path = os.path.join(str(tmp_path), "no_such_dir", "x.jsonl")
    tl.record(1, **{"loss/total": 1.0})
    tl.flush(1)                                   # must not raise
    assert tl.enabled is False


def test_header_and_summary_bracket_the_log(tmp_path):
    p = tmp_path / "t.jsonl"
    tl = TrainingLogger(p, run_name="r", config={"sigreg": True}, log_every=5)
    for step in range(1, 11):
        tl.record(step, **{"loss/total": 1.0})
        tl.maybe_flush(step)
    tl.close(10, status="budget_reached", memory=tl.memory_report())
    recs = [json.loads(l) for l in p.read_text().strip().split("\n")]
    assert recs[0]["type"] == "header" and recs[0]["config"]["sigreg"] is True
    assert recs[-1]["type"] == "summary" and recs[-1]["status"] == "budget_reached"
    assert "loss/total" in recs[-1]["lifetime"]


# ------------------------------------------------------------------- summariser
def write_synthetic(path, collapse):
    """A run where probe R^2 either holds or goes to zero as the loss falls."""
    tl = TrainingLogger(path, run_name="synthetic", config={"sigreg": not collapse},
                        log_every=10)
    n = 400
    for step in range(1, n + 1):
        frac = step / n
        tl.record(step, **{
            "loss/loss": 1.0 - 0.9 * frac,
            "loss/z_visual_loss": 0.6 - 0.55 * frac,
            "loss/sigreg_loss": 7.0 - 5.0 * frac,
            "grad/encoder.trunk/norm": 3.0,
            "grad/encoder.trunk/param_norm": 100.0,
            "grad/encoder.trunk/ratio": 0.03,
            "delta/encoder.trunk": 0.0 if collapse else 0.02,
        })
        if step % 100 == 0:
            r2 = 0.85 * (1 - frac) if collapse else 0.8
            tl.probe_latents(step, {"latent/probe_r2": max(r2, 0.0),
                                    "latent/eff_rank_frac": 0.1 if collapse else 0.4,
                                    "latent/std": 0.001 if collapse else 0.15})
        tl.maybe_flush(step)
    tl.close(n, status="completed")


def run_summariser(path, *args):
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "summarize_training_log.py"), str(path), *args],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_summariser_flags_the_collapse_signature(tmp_path):
    p = tmp_path / "collapsed.jsonl"
    write_synthetic(p, collapse=True)
    out = run_summariser(p)
    assert "COLLAPSE SIGNATURE" in out
    assert "NEVER MOVED" in out                   # trunk delta stayed at 0
    assert "collapse_warning" in out


def test_summariser_passes_a_healthy_run(tmp_path):
    p = tmp_path / "healthy.jsonl"
    write_synthetic(p, collapse=False)
    out = run_summariser(p)
    assert "COLLAPSE SIGNATURE" not in out
    assert "healthy run looks like" in out
    assert "NEVER MOVED" not in out


def test_summariser_output_size_is_independent_of_run_length(tmp_path):
    """Same digest length for a short and a long run: that is the whole point."""
    short, long = tmp_path / "s.jsonl", tmp_path / "l.jsonl"
    for path, n in ((short, 200), (long, 20_000)):
        tl = TrainingLogger(path, run_name="x", log_every=10)
        for step in range(1, n + 1):
            tl.record(step, **{"loss/loss": 1.0 / (1 + step)})
            tl.maybe_flush(step)
        tl.close(n)
    a = len(run_summariser(short).splitlines())
    b = len(run_summariser(long).splitlines())
    assert abs(a - b) <= 2, (a, b)


def test_summariser_survives_a_truncated_log(tmp_path):
    """A run killed mid-write leaves a partial final line."""
    p = tmp_path / "killed.jsonl"
    write_synthetic(p, collapse=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"type": "interval", "step": 99, "metr')
    out = run_summariser(p)
    assert "truncated line" in out


def test_summariser_reports_a_missing_summary(tmp_path):
    p = tmp_path / "running.jsonl"
    tl = TrainingLogger(p, run_name="x", log_every=5)
    for step in range(1, 21):
        tl.record(step, **{"loss/loss": 1.0})
        tl.maybe_flush(step)
    out = run_summariser(p)                       # never closed
    assert "NO SUMMARY RECORD" in out


def test_summariser_csv_export(tmp_path):
    p = tmp_path / "h.jsonl"
    write_synthetic(p, collapse=False)
    csv_path = tmp_path / "out.csv"
    out = run_summariser(p, "--csv", str(csv_path))
    assert csv_path.exists() and "wrote" in out
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("step,") and "loss/loss" in header


# ------------------------------------------------- input resolution / usability
def run_summariser_raw(*args):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "summarize_training_log.py"), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_unmatched_glob_explains_itself(tmp_path):
    """The shell leaves an unmatched glob literal; say something useful."""
    out = run_summariser_raw(str(tmp_path / "nope" / "*" / "*.jsonl"))
    assert out.returncode == 1
    assert "no telemetry logs found" in out.stderr
    assert "telemetry/train_<timestamp>.jsonl" in out.stderr
    assert "find . -name" in out.stderr          # actionable next command


def test_directory_is_searched_recursively(tmp_path):
    d = tmp_path / "run" / "telemetry"
    d.mkdir(parents=True)
    write_synthetic(d / "train_a.jsonl", collapse=False)
    out = run_summariser_raw(str(tmp_path))
    assert out.returncode == 0
    assert "train_a.jsonl" in out.stdout


def test_multiple_logs_are_each_digested(tmp_path):
    for name in ("a", "b"):
        d = tmp_path / name / "telemetry"
        d.mkdir(parents=True)
        write_synthetic(d / f"train_{name}.jsonl", collapse=False)
    out = run_summariser_raw(str(tmp_path))
    assert out.returncode == 0
    assert "2 logs matched" in out.stdout
    assert out.stdout.count("TRAINING TELEMETRY") == 2


def test_latest_picks_one(tmp_path):
    import time as _t
    for name in ("old", "new"):
        d = tmp_path / name / "telemetry"
        d.mkdir(parents=True)
        write_synthetic(d / f"train_{name}.jsonl", collapse=False)
        _t.sleep(0.02)
    out = run_summariser_raw(str(tmp_path), "--latest")
    assert out.returncode == 0
    assert out.stdout.count("TRAINING TELEMETRY") == 1
    assert "train_new.jsonl" in out.stdout


def test_header_only_log_is_not_an_error(tmp_path):
    """A run that died before the first flush must still be inspectable."""
    p = tmp_path / "train_x.jsonl"
    tl = TrainingLogger(p, run_name="barely started", log_every=500)
    tl.record(3, **{"loss/loss": 1.0})
    tl.event(3, "nonfinite", "something")
    out = run_summariser_raw(str(p))
    assert out.returncode == 0
    assert "no interval records yet" in out.stdout
    assert "nonfinite" in out.stdout             # the events still surface


# ------------------------------------------- verdict must not contradict itself
def write_run(path, r2_first, r2_last, rank_first, rank_last, std_first, std_last,
             loss_first=0.60, loss_last=0.02, n=400):
    """Synthetic run with prescribed endpoints for the verdict inputs."""
    tl = TrainingLogger(path, run_name="v", log_every=10)
    for step in range(1, n + 1):
        f = step / n
        tl.record(step, **{"loss/z_visual_loss": loss_first + (loss_last - loss_first) * f})
        if step % 50 == 0:
            tl.probe_latents(step, {
                "latent/probe_r2": r2_first + (r2_last - r2_first) * f,
                "latent/eff_rank_frac": rank_first + (rank_last - rank_first) * f,
                "latent/std": std_first + (std_last - std_first) * f,
            })
        tl.maybe_flush(step)
    tl.close(n, status="completed")


def test_real_collapse_still_reported(tmp_path):
    """Loss falls, R^2 falls, rank and std fall too -> genuine collapse."""
    p = tmp_path / "c.jsonl"
    write_run(p, r2_first=0.85, r2_last=0.00,
              rank_first=0.45, rank_last=0.13, std_first=0.39, std_last=0.002)
    out = run_summariser(p)
    assert "COLLAPSE SIGNATURE" in out
    assert "healthy run looks like" not in out


def test_reorganisation_is_not_called_collapse(tmp_path):
    """The observed case: R^2 halves but rank and std RISE. Not collapse."""
    p = tmp_path / "r.jsonl"
    write_run(p, r2_first=0.35, r2_last=0.05,
              rank_first=0.31, rank_last=0.38, std_first=0.35, std_last=0.39)
    out = run_summariser(p)
    assert "COLLAPSE SIGNATURE" not in out
    assert "NOT collapse" in out
    assert "reorganising" in out
    assert "healthy run looks like" not in out


def test_no_contradictory_verdict(tmp_path):
    """The bug: a FAIL threshold line printed alongside 'healthy run looks like'.

    R^2 0.2442 -> 0.1479 fails the >0.2 threshold but does not halve, which used
    to satisfy the 'healthy' branch as well.
    """
    p = tmp_path / "x.jsonl"
    write_run(p, r2_first=0.2442, r2_last=0.1479,
              rank_first=0.31, rank_last=0.38, std_first=0.35, std_last=0.39)
    out = run_summariser(p)
    assert "FAIL" in out                          # threshold check still fails
    assert "healthy run looks like" not in out    # ...and must not also say healthy
    assert "NOT collapse" in out


def test_healthy_run_still_reported(tmp_path):
    p = tmp_path / "h.jsonl"
    write_run(p, r2_first=0.60, r2_last=0.75,
              rank_first=0.40, rank_last=0.45, std_first=0.35, std_last=0.40)
    out = run_summariser(p)
    assert "healthy run looks like" in out
    assert "COLLAPSE SIGNATURE" not in out
    assert "NOT collapse" not in out


# ------------------------------------ the verdict must read the LIVE key names
def production_latent_keys():
    """Exactly the keys train.py feeds to probe_latents, derived not hardcoded.

    models.diagnostics.latent_diagnostics(prefix="val_") emits e.g.
    'val_latent_eff_rank_frac'; train.py then strips only the 'val_' prefix and
    prepends 'latent/', giving 'latent/latent_eff_rank_frac'. Deriving them here
    means this test cannot drift from the trainer.
    """
    import torch as _t

    from models.diagnostics import latent_diagnostics

    diag = latent_diagnostics(_t.randn(8, 4, 8), state=_t.randn(8, 4, 3), prefix="val_")
    return ["latent/" + k.replace("val_", "", 1) for k in diag]


def test_production_key_names_are_what_we_expect():
    keys = production_latent_keys()
    assert "latent/probe_r2" in keys
    assert "latent/latent_eff_rank_frac" in keys      # NOT latent/eff_rank_frac
    assert "latent/latent_std" in keys                # NOT latent/std


def test_verdict_uses_the_production_keys(tmp_path):
    """Regression: the fix relied on a fallback, and the tests only covered the
    other branch. Write the real key names and check the verdict still fires."""
    p = tmp_path / "prod.jsonl"
    tl = TrainingLogger(p, run_name="prod", log_every=10)
    n = 400
    for step in range(1, n + 1):
        f = step / n
        tl.record(step, **{"loss/z_visual_loss": 0.60 - 0.58 * f})
        if step % 50 == 0:
            tl.probe_latents(step, {
                # the observed live run: R^2 down, rank and std UP
                "latent/probe_r2": 0.2442 - 0.0963 * f,
                "latent/latent_eff_rank_frac": 0.31 + 0.07 * f,
                "latent/latent_std": 0.35 + 0.04 * f,
            })
        tl.maybe_flush(step)
    tl.close(n, status="completed")

    out = run_summariser(p)
    assert "NOT collapse" in out
    assert "reorganising" in out
    assert "healthy run looks like" not in out
    assert "COLLAPSE SIGNATURE" not in out


def test_production_keys_still_detect_real_collapse(tmp_path):
    p = tmp_path / "prodc.jsonl"
    tl = TrainingLogger(p, run_name="prodc", log_every=10)
    n = 400
    for step in range(1, n + 1):
        f = step / n
        tl.record(step, **{"loss/z_visual_loss": 0.60 - 0.58 * f})
        if step % 50 == 0:
            tl.probe_latents(step, {
                "latent/probe_r2": 0.85 * (1 - f),
                "latent/latent_eff_rank_frac": 0.45 - 0.32 * f,   # ends 0.13 < 0.2
                "latent/latent_std": 0.39 * (1 - f) + 0.0005,
            })
        tl.maybe_flush(step)
    tl.close(n, status="completed")

    out = run_summariser(p)
    assert "COLLAPSE SIGNATURE" in out
    assert "NOT collapse" not in out


# --------------------------------- sparse metrics must not alias out of the table
def write_sparse_latents(path, n_intervals=475, log_every=200, diag_every=500):
    """Mirror the live run: telemetry every 200 steps, diagnostics every 500.

    Latent metrics therefore exist in only ~2 of every 5 interval records, which
    is what let index-based downsampling select nothing but empty rows.
    """
    tl = TrainingLogger(path, run_name="sparse", log_every=log_every)
    for step in range(1, n_intervals * log_every + 1):
        tl.record(step, **{"loss/z_visual_loss": 1.0 / (1 + step / 1000)})
        if step % diag_every == 0:
            tl.probe_latents(step, {
                "latent/probe_r2": 0.30,
                "latent/latent_eff_rank_frac": 0.35,
                "latent/latent_std": 0.38,
            })
        tl.maybe_flush(step)
    tl.close(n_intervals * log_every, status="completed")


def test_sparse_latent_table_is_not_all_dashes(tmp_path):
    """The regression: 475 intervals, 22 rows, every row printed '-'."""
    p = tmp_path / "sparse.jsonl"
    write_sparse_latents(p)
    out = run_summariser(p, "--metrics", "latent")
    latent_block = out.split("## Latent health")[1].split("##")[0]
    body = [l for l in latent_block.splitlines()
            if l.strip() and "step" not in l and "intervals carry" not in l]
    assert body, "latent table produced no rows at all"
    populated_rows = [l for l in body if "0.3" in l]
    assert populated_rows, f"every row was empty:\n{latent_block}"


def test_sparse_table_reports_how_many_intervals_carry_the_metrics(tmp_path):
    p = tmp_path / "sparse2.jsonl"
    write_sparse_latents(p)
    out = run_summariser(p, "--metrics", "latent")
    assert "intervals carry these metrics" in out


@pytest.mark.parametrize("n_intervals", [97, 200, 468, 475, 619])
def test_no_aliasing_at_any_run_length(n_intervals):
    """It worked at 468 intervals and broke at 475; sweep several lengths."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.jsonl")
        write_sparse_latents(p, n_intervals=n_intervals)
        out = run_summariser(p, "--metrics", "latent")
        block = out.split("## Latent health")[1].split("##")[0]
        assert "0.3" in block, f"aliased to empty rows at {n_intervals} intervals"


def test_dense_metrics_table_unchanged(tmp_path):
    """A section present in every interval must not gain the note line."""
    p = tmp_path / "dense.jsonl"
    tl = TrainingLogger(p, run_name="dense", log_every=10)
    for step in range(1, 201):
        tl.record(step, **{"loss/z_visual_loss": 1.0 / (1 + step)})
        tl.maybe_flush(step)
    tl.close(200)
    out = run_summariser(p, "--metrics", "loss")
    assert "intervals carry these metrics" not in out
