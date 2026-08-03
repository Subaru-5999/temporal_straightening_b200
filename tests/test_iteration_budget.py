"""Tests for the training.max_iterations knob (iteration_budget.IterationBudget).

Covers the arithmetic and a simulation of train()/run()'s stop semantics, so the
mid-epoch break, the resume path and the "never extends a run" guarantee are all
checked without torch, hydra, accelerate or a dataset.

Run:  pytest tests/test_iteration_budget.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iteration_budget import IterationBudget

PUSHT_ITERS_PER_EPOCH = 61_929   # observed in train_pusht_channel_on.log, batch 32
PUSHT_PAPER_EPOCHS = 2           # paper App. A.3
PUSHT_PAPER_ITERS = 123_858      # 61,929 x 2


# ------------------------------------------------------------------ arithmetic
def test_disabled_by_default_matches_epoch_behaviour():
    b = IterationBudget(iters_per_epoch=100, epochs=3)
    assert not b.enabled
    assert b.effective_total == 300
    assert b.remaining(0) is None
    assert not b.reached(10**9)          # never stops early when disabled


def test_pusht_paper_budget():
    b = IterationBudget(PUSHT_ITERS_PER_EPOCH, PUSHT_PAPER_EPOCHS, PUSHT_PAPER_ITERS)
    assert b.epoch_bounded_total == PUSHT_PAPER_ITERS
    assert b.effective_total == PUSHT_PAPER_ITERS
    assert b.epochs_needed == PUSHT_PAPER_EPOCHS
    assert b.is_reachable
    assert not b.reached(PUSHT_PAPER_ITERS - 1)
    assert b.reached(PUSHT_PAPER_ITERS)
    assert b.remaining(PUSHT_PAPER_ITERS) == 0


def test_paper_budget_reachable_when_epochs_raised():
    # the intended use for a modified objective: keep the paper's step budget
    # while allowing more epochs so the cap, not the epoch count, ends the run
    b = IterationBudget(PUSHT_ITERS_PER_EPOCH, epochs=3, max_iterations=PUSHT_PAPER_ITERS)
    assert b.is_reachable
    assert b.effective_total == PUSHT_PAPER_ITERS
    assert b.epochs_needed == 2


def test_cap_never_extends_a_run():
    b = IterationBudget(iters_per_epoch=100, epochs=2, max_iterations=10_000)
    assert not b.is_reachable
    assert b.effective_total == 200          # epochs still bound it
    assert b.epochs_needed == 100
    assert "WARNING" in b.describe()


def test_mid_epoch_stop_is_exact():
    b = IterationBudget(iters_per_epoch=100, epochs=5, max_iterations=250)
    assert b.epochs_needed == 3              # stops halfway through epoch 3
    assert b.effective_total == 250


def test_resume_estimate_for_legacy_checkpoints():
    b = IterationBudget(PUSHT_ITERS_PER_EPOCH, 2, PUSHT_PAPER_ITERS)
    assert b.resume_estimate(0) == 0
    assert b.resume_estimate(1) == PUSHT_ITERS_PER_EPOCH
    assert b.remaining(b.resume_estimate(1)) == PUSHT_ITERS_PER_EPOCH


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonpositive_cap(bad):
    with pytest.raises(ValueError):
        IterationBudget(100, 2, bad)


@pytest.mark.parametrize("kwargs", [
    dict(iters_per_epoch=0, epochs=2),
    dict(iters_per_epoch=10, epochs=0),
])
def test_rejects_degenerate_shapes(kwargs):
    with pytest.raises(ValueError):
        IterationBudget(**kwargs)


# ------------------------------------------- simulation of train()/run() wiring
def simulate(iters_per_epoch, epochs, max_iterations, start_epoch=0, start_iter=0):
    """Mirror Trainer.run()/train(): increment per step, break on budget.

    Returns (total_steps, epochs_entered, stopped_on_budget, saved_final).
    """
    b = IterationBudget(iters_per_epoch, epochs, max_iterations)
    global_iter = start_iter
    stop_requested = False
    epochs_entered = 0
    saved_final = False

    for _ in range(start_epoch + 1, start_epoch + 1 + epochs):   # run()
        epochs_entered += 1
        for _ in range(iters_per_epoch):                         # train()
            global_iter += 1
            if b.reached(global_iter):
                stop_requested = True
                break
        saved_final = True              # run() saves after every epoch body
        if stop_requested:
            break
    return global_iter, epochs_entered, stop_requested, saved_final


def test_simulation_stops_exactly_on_budget():
    steps, epochs_entered, stopped, saved = simulate(100, 5, 250)
    assert (steps, epochs_entered, stopped, saved) == (250, 3, True, True)


def test_simulation_disabled_runs_all_epochs():
    steps, epochs_entered, stopped, _ = simulate(100, 5, None)
    assert (steps, epochs_entered, stopped) == (500, 5, False)


def test_simulation_pusht_paper_budget():
    steps, epochs_entered, stopped, saved = simulate(
        PUSHT_ITERS_PER_EPOCH, PUSHT_PAPER_EPOCHS, PUSHT_PAPER_ITERS
    )
    assert steps == PUSHT_PAPER_ITERS
    assert epochs_entered == PUSHT_PAPER_EPOCHS
    assert stopped and saved


def test_simulation_resume_does_not_double_spend():
    """Resuming after 1 completed epoch must add only the remaining steps."""
    steps, epochs_entered, stopped, _ = simulate(
        PUSHT_ITERS_PER_EPOCH, epochs=2, max_iterations=PUSHT_PAPER_ITERS,
        start_epoch=1, start_iter=PUSHT_ITERS_PER_EPOCH,
    )
    assert steps == PUSHT_PAPER_ITERS       # not 123,858 + 61,929
    assert epochs_entered == 1
    assert stopped
