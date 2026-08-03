"""Iteration-budget bookkeeping for paper-exact training runs.

The paper specifies training length in *epochs* (App. A: Wall/PointMaze 20,
PushT 2). When the objective or the set of trainable parameters changes, it is
useful to pin the run to the same number of *optimizer steps* instead, so two
variants are compared at an identical compute/update budget rather than at an
identical number of passes over a possibly-resampled dataset.

`training.max_iterations` is that knob. It is a hard cap on the total number of
optimizer steps for the whole run, counted across epochs and preserved across
resumes. It never *extends* training: the epoch count remains an upper bound.

Deliberately stdlib-only so it can be imported and tested without torch,
hydra, accelerate or a dataset.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IterationBudget:
    """Total-optimizer-step budget for a run.

    Args:
        iters_per_epoch: batches per epoch per process, i.e. ``len(dataloader)``
            after ``accelerator.prepare`` (one optimizer step per batch).
        epochs: ``training.epochs``; always an upper bound on the run.
        max_iterations: ``training.max_iterations``; ``None`` disables the cap
            and restores pure epoch-bounded behaviour.
    """

    iters_per_epoch: int
    epochs: int
    max_iterations: Optional[int] = None

    def __post_init__(self):
        if self.iters_per_epoch <= 0:
            raise ValueError(f"iters_per_epoch must be > 0, got {self.iters_per_epoch}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError(
                f"training.max_iterations must be > 0 or null, got {self.max_iterations}"
            )

    @property
    def enabled(self) -> bool:
        return self.max_iterations is not None

    @property
    def epoch_bounded_total(self) -> int:
        """Steps the run would take on the epoch count alone."""
        return self.iters_per_epoch * self.epochs

    @property
    def effective_total(self) -> int:
        """Steps the run will actually take (whichever bound binds first)."""
        if not self.enabled:
            return self.epoch_bounded_total
        return min(self.max_iterations, self.epoch_bounded_total)

    @property
    def epochs_needed(self) -> int:
        """Epochs required to reach `max_iterations` (ceil division)."""
        if not self.enabled:
            return self.epochs
        return -(-self.max_iterations // self.iters_per_epoch)

    @property
    def is_reachable(self) -> bool:
        """False when `epochs` is too small for the cap to ever be hit."""
        return (not self.enabled) or self.max_iterations <= self.epoch_bounded_total

    def reached(self, global_iter: int) -> bool:
        """True once `global_iter` optimizer steps have been taken."""
        return self.enabled and global_iter >= self.max_iterations

    def remaining(self, global_iter: int) -> Optional[int]:
        if not self.enabled:
            return None
        return max(0, self.max_iterations - global_iter)

    def resume_estimate(self, epoch: int) -> int:
        """Step count to assume when resuming a ckpt saved before this knob existed.

        Old checkpoints carry `epoch` but no `global_iter`; `epoch` counts
        *completed* epochs, so the step count is that many full epochs.
        """
        return max(0, epoch) * self.iters_per_epoch

    def describe(self) -> str:
        lines = [
            "Iteration budget:",
            f"  iters_per_epoch      : {self.iters_per_epoch:,}",
            f"  epochs               : {self.epochs}",
            f"  epoch-bounded total  : {self.epoch_bounded_total:,}",
            f"  training.max_iterations: "
            f"{'null (disabled)' if not self.enabled else format(self.max_iterations, ',')}",
            f"  effective total steps: {self.effective_total:,}",
        ]
        if self.enabled and not self.is_reachable:
            lines.append(
                f"  WARNING: cap {self.max_iterations:,} needs {self.epochs_needed} "
                f"epochs but training.epochs={self.epochs}; the run will stop at "
                f"{self.epoch_bounded_total:,} steps, SHORT of the budget. "
                f"Raise training.epochs to at least {self.epochs_needed}."
            )
        elif self.enabled and self.max_iterations < self.epoch_bounded_total:
            lines.append(
                f"  cap binds first: stopping mid-epoch "
                f"{self.epochs_needed} of {self.epochs}."
            )
        return "\n".join(lines)
