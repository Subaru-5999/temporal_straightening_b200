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
