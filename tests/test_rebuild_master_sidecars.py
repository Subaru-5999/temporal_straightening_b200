"""rebuild_master must ignore sidecar files in results/.

Regression: reproduce_table1.py writes results/<run>.timing.json holding per-arm
wall-clock seconds. rebuild_master globbed results/*.json, loaded it as a run
record, and died with KeyError: 'run' after a full 3-seed evaluation had already
completed -- losing the master table for a run that took ~50 minutes to produce.
"""

import json
import os

import pytest

import summarize_run


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(summarize_run, "RESULTS_DIR", str(d))
    monkeypatch.chdir(tmp_path)
    return d


def _record(name):
    return {
        "run": name,
        "label": name,
        "open_loop": {"seeds": [30.0], "mean": 30.0, "std": 0.0, "n": 1, "paper": None},
        "mpc": {"seeds": [40.0], "mean": 40.0, "std": 0.0, "n": 1, "paper": None},
        "cem_open_loop": {"seeds": [], "mean": None, "std": None, "n": 0, "paper": None},
        "timing": None,
    }


def test_timing_sidecar_does_not_crash_rebuild(results_dir):
    (results_dir / "myrun.json").write_text(json.dumps(_record("myrun")))
    # the sidecar: no "run" key, which is what broke it
    (results_dir / "myrun.timing.json").write_text(
        json.dumps({"gd": [76.2], "gd_mpc": [1545.8]})
    )
    summarize_run.rebuild_master()          # must not raise
    md = results_dir / "table1_reproduction.md"
    assert md.exists() and "myrun" in md.read_text()


def test_unrelated_json_without_a_run_key_is_dropped(results_dir):
    (results_dir / "myrun.json").write_text(json.dumps(_record("myrun")))
    (results_dir / "notes.json").write_text(json.dumps({"anything": 1}))
    (results_dir / "alist.json").write_text(json.dumps([1, 2, 3]))
    summarize_run.rebuild_master()
    assert "myrun" in (results_dir / "table1_reproduction.md").read_text()


def test_seed_variants_are_still_excluded(results_dir):
    (results_dir / "myrun.json").write_text(json.dumps(_record("myrun")))
    (results_dir / "myrun_seed7.json").write_text(json.dumps(_record("myrun_seed7")))
    (results_dir / "myrun_trainseed.json").write_text(json.dumps(_record("myrun_ts")))
    summarize_run.rebuild_master()
    text = (results_dir / "table1_reproduction.md").read_text()
    assert "myrun" in text
    assert "myrun_seed7" not in text
    assert "myrun_ts" not in text


# --------------------------------------------------------------------------
# Run-name nesting. plan.py writes its logs to "<run>_gH<H>_<goal_source>/", and
# run names nest: "..._e2e" is a strict prefix of "..._e2e_gp1e0". A "{name}_*"
# glob made the shorter run absorb the longer one's logs.json and report 6 pooled
# seeds as if they belonged to one run -- a wrong number that would have gone
# straight into the paper.
# --------------------------------------------------------------------------

SHORT = "pusht_x_sgFalse_lr1e-05_sig1e-1_e2e"
LONG = SHORT + "_gp1e0"


def _write_logs(root, dirname, values, key="final_eval/success_rate"):
    d = os.path.join(root, dirname, "sub")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "logs.json"), "w") as f:
        for v in values:
            f.write(json.dumps({key: v}) + "\n")


@pytest.fixture
def plan_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_nested_run_names_do_not_pool_seeds(plan_outputs):
    _write_logs("plan_outputs_gd", f"{SHORT}_gH25_dset", [0.12, 0.14, 0.14])
    _write_logs("plan_outputs_gd", f"{LONG}_gH25_dset", [0.38, 0.28, 0.24])

    short_vals = summarize_run.read_success_rates("plan_outputs_gd", SHORT)
    long_vals = summarize_run.read_success_rates("plan_outputs_gd", LONG)

    assert sorted(short_vals) == [12.0, 14.0, 14.0], short_vals
    assert sorted(long_vals) == [24.0, 28.0, 38.0], long_vals
    assert len(short_vals) == 3, "the shorter name absorbed the longer run's logs"


def test_diagnostics_in_a_subdirectory_are_not_discovered(results_dir):
    """discover_runs is name-based and only globs results/*.json, not subdirs.

    So diagnostic dumps belong in results/diagnostics/ -- putting them at the top
    level makes --all emit a "no logs found" line per diagnostic.
    """
    (results_dir / "myrun.json").write_text(json.dumps(_record("myrun")))
    diag = results_dir / "diagnostics"
    diag.mkdir()
    (diag / "metric_alignment_method.json").write_text(json.dumps({"probe_r2": 0.5}))
    (diag / "rollout_drift_method.json").write_text(json.dumps({"visual": {}}))
    names = summarize_run.discover_runs()
    assert "myrun" in names
    assert "metric_alignment_method" not in names
    assert "rollout_drift_method" not in names
