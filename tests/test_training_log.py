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
