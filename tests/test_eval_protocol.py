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
