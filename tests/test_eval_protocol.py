"""Tests for reproduce_table1's run-name -> eval-protocol mapping.

The five tracked Table-1 cells must keep their exact (alpha, mpc_mode). New
objective variants (SIGReg / end-to-end, which carry a _sig..._e2e suffix and
sgFalse instead of sgTrue) must resolve to their environment's protocol rather
than being skipped as unknown.

reproduce_table1 sets a pile of os.environ defaults and imports summarize_run at
module scope but no third-party packages, so it imports fine here.

Run:  pytest tests/test_eval_protocol.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reproduce_table1 as rt

UMAZE_BASELINE = "umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
PUSHT_BASELINE = "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
PUSHT_E2E = "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgFalse_lr1e-05_sig1e-1_e2e"
PUSHT_GATE1 = "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05_e2e"


# --------------------------------------------------- the five cells are untouched
def test_all_tracked_cells_resolve_to_themselves():
    for name in rt.ORDER:
        assert rt.base_cell(name) == name
        assert rt.eval_protocol(name) == rt.CFG[name]


def test_tracked_protocols_match_sec_5_3():
    """PushT uses proprio + a staged MPC objective; the mazes use images only."""
    for name in rt.ORDER:
        alpha, mode = rt.eval_protocol(name)
        if name.startswith("pusht"):
            assert (alpha, mode) == (1, "staged")
        else:
            assert (alpha, mode) == (0, "all")


def test_training_seed_variants_still_inherit_their_cell():
    assert rt.base_cell(PUSHT_BASELINE + "_seed2") == PUSHT_BASELINE
    assert rt.eval_protocol(PUSHT_BASELINE + "_seed2") == (1, "staged")


# ------------------------------------------------------- new objective variants
def test_end_to_end_sigreg_variant_is_evaluated_not_skipped():
    """The regression this fixes: exact-match only would return None here."""
    assert rt.base_cell(PUSHT_E2E) is None          # not a tracked cell
    assert rt.eval_protocol(PUSHT_E2E) == (1, "staged")


def test_gate1_control_variant_resolves():
    assert rt.eval_protocol(PUSHT_GATE1) == (1, "staged")


def test_umaze_variant_resolves_to_the_maze_protocol():
    umaze_e2e = UMAZE_BASELINE.replace("sgTrue", "sgFalse") + "_sig1e-1_e2e"
    assert rt.eval_protocol(umaze_e2e) == (0, "all")


@pytest.mark.parametrize("env,expected", [
    ("umaze", (0, "all")),
    ("medium", (0, "all")),
    ("wall", (0, "all")),
    ("pusht", (1, "staged")),
])
def test_every_env_prefix_has_a_protocol(env, expected):
    assert rt.eval_protocol(f"{env}_whatever_config_string") == expected


def test_unknown_env_is_still_refused():
    """Fail closed: an unrecognised env must not silently get a wrong objective."""
    assert rt.eval_protocol("atari_something_sgTrue_lr1e-05") is None
    assert rt.eval_protocol("") is None


def test_env_defaults_agree_with_the_tracked_cells():
    """No contradiction between the two sources of truth."""
    for name, protocol in rt.CFG.items():
        env = name.split("_", 1)[0]
        assert rt.ENV_DEFAULTS[env] == protocol, name


# ------------------------------------------------- CEM arm + planning-time capture
import json

import summarize_run as sr


def test_plan_roots_cover_the_three_arms():
    assert set(rt.PLAN_ROOTS) == {"gd", "gd_mpc", "cem"}
    for arm, (root, cfg, extra) in rt.PLAN_ROOTS.items():
        assert root.startswith("plan_outputs_")
        assert cfg.endswith(".yaml")
        assert isinstance(extra, list)


def test_cem_arm_is_open_loop_terminal_objective():
    """Table 4: open-loop uses the terminal objective and executes 25 actions.
    conf/plan_cem.yaml already sets max_iter=1 / n_taken_actions=25, so the arm
    only needs to pin objective.mode=last."""
    _, cfg, extra = rt.PLAN_ROOTS["cem"]
    assert cfg == "plan_cem.yaml"
    assert "objective.mode=last" in extra


def test_gd_arm_carries_no_extra_overrides():
    """The GD arms must stay byte-identical to the reproduced Table 1."""
    assert rt.PLAN_ROOTS["gd"][2] == []
    assert rt.PLAN_ROOTS["gd_mpc"][2] == []


@pytest.mark.parametrize("line,expected", [
    ("[timing] perform_planning_s=123.456", 123.456),
    ("prefix [timing] perform_planning_s=0.5 suffix", 0.5),
    ("[timing]perform_planning_s=7", 7.0),
])
def test_timing_line_is_parsed(line, expected):
    m = rt.TIMING_RE.search(line)
    assert m and float(m.group(1)) == expected


@pytest.mark.parametrize("line", [
    "[timing] setup_model_s=12.3",           # a different timer
    "[timing] total_planning_main_s=99.9",   # must not be mistaken for it
    "perform_planning_s=5",                  # no [timing] tag
])
def test_other_timing_lines_are_ignored(line):
    assert rt.TIMING_RE.search(line) is None


def test_read_timing_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "RESULTS_DIR", str(tmp_path))
    assert sr.read_timing("nope") == {}


def test_read_timing_averages_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "RESULTS_DIR", str(tmp_path))
    (tmp_path / "r.timing.json").write_text(
        json.dumps({"gd": [10.0, 20.0, 30.0], "cem": [100.0, 200.0]}))
    t = sr.read_timing("r")
    assert t["gd"]["mean_s"] == pytest.approx(20.0)
    assert t["gd"]["n"] == 3
    assert t["cem"]["mean_s"] == pytest.approx(150.0)


def test_read_timing_survives_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "RESULTS_DIR", str(tmp_path))
    (tmp_path / "r.timing.json").write_text("{not json")
    assert sr.read_timing("r") == {}


def test_read_timing_drops_nulls(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "RESULTS_DIR", str(tmp_path))
    (tmp_path / "r.timing.json").write_text(json.dumps({"gd": [None], "cem": [5.0]}))
    t = sr.read_timing("r")
    assert "gd" not in t and t["cem"]["mean_s"] == 5.0


def test_discover_runs_includes_the_tracked_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sr, "RESULTS_DIR", str(tmp_path / "results"))
    assert set(sr.discover_runs()) >= set(sr.PAPER)


def test_discover_runs_finds_objective_variants(tmp_path, monkeypatch):
    """The regression: `--all` iterated PAPER only, so variant runs were skipped."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sr, "RESULTS_DIR", str(tmp_path / "results"))
    variant = "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgFalse_lr1e-05_sig1e-1_e2e"
    (tmp_path / "plan_outputs_gd" / f"{variant}_gH25_dset").mkdir(parents=True)
    (tmp_path / "plan_outputs_cem" / f"{variant}_gH25_dset").mkdir(parents=True)
    found = sr.discover_runs()
    assert variant in found
    assert found.count(variant) == 1          # de-duplicated across roots


def test_discover_runs_ignores_bookkeeping_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(sr, "RESULTS_DIR", str(res))
    for junk in ("table1_reproduction.json", "x.timing.json", "y_trainseed.json",
                 "z_seed2.json"):
        (res / junk).write_text("{}")
    (res / "real_run.json").write_text("{}")
    found = sr.discover_runs()
    assert "real_run" in found
    for bad in ("table1_reproduction", "x", "y_trainseed", "z_seed2"):
        assert bad not in found
