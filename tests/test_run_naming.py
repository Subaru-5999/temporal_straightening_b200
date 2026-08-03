"""Tests for run_naming.variant_tag.

The critical property is backward compatibility: with the original defaults the
tag must be exactly '', so existing checkpoint directories -- and therefore the
Table-1 reproduction harness that looks them up by name -- keep working.

Run:  pytest tests/test_run_naming.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_naming import fmt_coeff, truthy, variant_tag

DEFAULTS = dict(sigreg=False, sigreg_coeff=0.0, freeze_backbone=True,
                curv_on="features")

# the five run names the existing eval driver (reproduce_table1.py) knows
EXISTING_RUNS = [
    "umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05",
    "umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06",
    "umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06",
    "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
]


def test_defaults_produce_no_suffix():
    assert variant_tag(**DEFAULTS) == ""


def test_existing_run_names_are_unchanged():
    tag = variant_tag(**DEFAULTS)
    for name in EXISTING_RUNS:
        assert name + tag == name


@pytest.mark.parametrize("falsey", [False, "False", "false", None, "None", "null", "", 0])
def test_string_and_bool_falsey_are_both_handled(falsey):
    """OmegaConf interpolation can deliver either a bool or its string form."""
    assert variant_tag(sigreg=falsey, sigreg_coeff=0.0, freeze_backbone=True,
                       curv_on="features") == ""
    assert variant_tag(sigreg=False, sigreg_coeff=0.0, freeze_backbone=falsey,
                       curv_on="features") == "_e2e"


def test_sigreg_on_with_zero_coeff_is_still_a_no_op():
    """A zero weight means the term contributes nothing, so don't rename the run."""
    assert variant_tag(sigreg=True, sigreg_coeff=0.0, freeze_backbone=True,
                       curv_on="features") == ""


def test_full_variant():
    assert variant_tag(True, 0.1, False, "features") == "_sig1e-1_e2e"


def test_curv_on_velocity_ablation():
    assert variant_tag(True, 0.1, False, "velocity") == "_sig1e-1_e2e_curvvel"


def test_negative_control_is_distinguishable():
    """unfrozen but no SIGReg -- gate 1. Must not collide with the baseline."""
    assert variant_tag(False, 0.0, False, "features") == "_e2e"


def test_variants_are_pairwise_distinct():
    combos = [
        (False, 0.0, True, "features"),
        (False, 0.0, False, "features"),
        (True, 0.1, True, "features"),
        (True, 0.1, False, "features"),
        (True, 0.01, False, "features"),
        (True, 0.1, False, "velocity"),
    ]
    tags = [variant_tag(*c) for c in combos]
    assert len(set(tags)) == len(tags), tags


@pytest.mark.parametrize("value,expected", [
    (0.1, "1e-1"), (0.01, "1e-2"), (1.0, "1e0"), (0.0, "0"), (1e-5, "1e-5"),
])
def test_fmt_coeff(value, expected):
    assert fmt_coeff(value) == expected


def test_fmt_coeff_survives_junk():
    assert fmt_coeff("abc") == "abc"


def test_tag_is_filesystem_safe():
    for combo in [(True, 0.1, False, "velocity"), (True, 0.01, False, "features")]:
        tag = variant_tag(*combo)
        assert all(c.isalnum() or c in "_-" for c in tag), tag


@pytest.mark.parametrize("v", [True, "True", "yes", 1, "1"])
def test_truthy(v):
    assert truthy(v)
