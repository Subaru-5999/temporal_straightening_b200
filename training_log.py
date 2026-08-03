"""Bounded-memory training telemetry: a replayable traceback of a long run.

We cannot observe the latent space directly, so the alternative to guessing is to
record, every N steps, the small set of scalars that actually distinguish a
healthy run from a collapsing one -- and to do it in memory that does not grow
with the length of the run.

Design
------
O(1) memory per metric. Raw per-step values are never retained. Each metric is
folded into `OnlineStats`, which keeps Welford's running mean/variance plus
min/max/last -- six floats, regardless of how many steps you feed it. See
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance . This is the
same substitution NVIDIA's telemetry makes ("replaces ... MetricsAccumulator
which stores all raw samples ... O(1) memory usage"). Only two things are
retained as history, both hard-capped: a `deque(maxlen=)` of recent anomaly
events, and a `deque(maxlen=)` of recent interval digests for spike context.

O(1) disk per interval. Every `log_every` steps one JSON object is appended and
the interval accumulators reset. For the PushT budget of 123,858 steps at
log_every=200 that is 620 interval records, a few MB. Predictable, greppable,
and small enough to hand to a reader in full.

What is recorded, and why each one earns its place
-------------------------------------------------
loss/*        every term separately (pred, visual, proprio, curvature, sigreg,
              raw and scaled). A falling total tells you nothing about which
              term is doing the work, or which one quietly went to zero.
latent/*      std across (batch, time), effective rank, curvature cosine,
              linear-probe R^2. The collapse detectors. The prediction loss
              falling while these fall is the failure mode, not success.
grad/*        per-module-group gradient norm, parameter norm, and their ratio.
              Standard practice for run health; the ratio is what controls the
              per-layer effective learning rate, so it is the quantity that
              tells you whether backbone_lr is sane.
delta/*       measured L2 movement of a sampled weight tensor per group since
              the previous probe. Answers "is the trunk actually moving?" with
              a number rather than an assumption, using a fixed element budget.
events        NaN/Inf, loss spikes relative to the running mean, and collapse
              threshold crossings -- with the step number, so you can locate
              the moment something turned.

Stdlib + optional torch: `torch` is imported lazily and only inside the module
probes, so this file is importable and testable without it.
"""

import json
import math
import os
import time
from collections import deque

SCHEMA_VERSION = 1


class OnlineStats:
    """Welford running mean/variance plus min/max/last. Six floats, O(1) memory."""

    __slots__ = ("n", "mean", "_m2", "min", "max", "last", "n_nonfinite")

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.last = float("nan")
        self.n_nonfinite = 0

    def update(self, x):
        x = float(x)
        self.last = x
        if not math.isfinite(x):
            self.n_nonfinite += 1
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    @property
    def var(self):
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(max(0.0, self.var))

    def as_dict(self, round_to=6):
        if self.n == 0:
            return {"n": 0, "nonfinite": self.n_nonfinite}
        r = lambda v: round(v, round_to)          # noqa: E731
        out = {"n": self.n, "mean": r(self.mean), "std": r(self.std),
               "min": r(self.min), "max": r(self.max), "last": r(self.last)}
        if self.n_nonfinite:
            out["nonfinite"] = self.n_nonfinite
        return out


class _DeltaTracker:
    """Measured movement of a few sampled weight tensors, on a fixed budget.

    Keeping a copy of every parameter would cost as much as the model. Instead we
    clone one representative tensor per module group, subject to a global element
    budget, and report ||w_now - w_at_last_probe|| for each. Bounded memory, and
    it is a measurement rather than a proxy.
    """

    def __init__(self, element_budget=2_000_000):
        self.element_budget = int(element_budget)
        self._snap = {}          # group -> (param_ref, cloned_tensor)
        self._used = 0

    def _pick(self, module):
        best = None
        for name, p in module.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                if best is None or p.numel() > best[1].numel():
                    best = (name, p)
        if best is None:
            for name, p in module.named_parameters():
                if p.requires_grad:
                    return name, p
        return best if best else (None, None)

    def probe(self, group, module):
        """Return (delta_l2, tensor_name) or (None, None) if not tracked."""
        if group not in self._snap:
            name, p = self._pick(module)
            if p is None:
                return None, None
            if self._used + p.numel() > self.element_budget:
                return None, None
            self._snap[group] = (name, p, p.detach().float().cpu().clone())
            self._used += p.numel()
            return 0.0, name
        name, p, prev = self._snap[group]
        cur = p.detach().float().cpu()
        delta = float((cur - prev).norm().item())
        self._snap[group] = (name, p, cur.clone())
        return delta, name

    @property
    def elements_tracked(self):
        return self._used


class TrainingLogger:
    """Append-only JSONL telemetry with bounded memory and bounded output.

    Usage from a training loop::

        tl = TrainingLogger("run.jsonl", run_name=..., config=...)
        ...
        tl.record(step, **{"loss/total": 0.4, "loss/sigreg": 2.1})
        tl.probe_modules(step, {"encoder.trunk": trunk_mod, ...}, lrs)
        tl.probe_latents(step, {"latent/probe_r2": 0.74, ...})
        tl.maybe_flush(step)
        ...
        tl.close(status="completed")
    """

    def __init__(self, path, run_name="", config=None, log_every=200,
                 ring_events=256, ring_intervals=32, spike_sigma=8.0,
                 spike_min_n=200, delta_element_budget=2_000_000,
                 collapse_thresholds=None, enabled=True):
        self.path = str(path)
        self.run_name = run_name
        self.log_every = max(1, int(log_every))
        self.spike_sigma = float(spike_sigma)
        self.spike_min_n = int(spike_min_n)
        self.enabled = bool(enabled)

        # --- the only two histories, both hard-capped ---
        self.events = deque(maxlen=int(ring_events))
        self.recent_intervals = deque(maxlen=int(ring_intervals))

        # --- O(1) accumulators ---
        self._interval = {}      # key -> OnlineStats, reset every flush
        self._lifetime = {}      # key -> OnlineStats, never reset (for spikes)
        self._delta = _DeltaTracker(delta_element_budget)

        self.collapse_thresholds = collapse_thresholds or {
            "latent/probe_r2": ("<", 0.10),
            "latent/eff_rank_frac": ("<", 0.20),
            "latent/std": ("<", 1e-3),
        }

        self._n_intervals = 0
        self._n_events = 0
        self._last_flush_step = 0
        self._last_flush_time = time.time()
        self._t0 = time.time()
        self._closed = False

        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            self._write({
                "type": "header", "schema": SCHEMA_VERSION, "run": run_name,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "log_every": self.log_every, "config": config or {},
            })

    # ------------------------------------------------------------------ plumbing
    def _write(self, obj):
        if not self.enabled:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, default=str, sort_keys=True) + "\n")
        except Exception:
            self.enabled = False        # telemetry must never kill a training run

    def _stats(self, table, key):
        st = table.get(key)
        if st is None:
            st = table[key] = OnlineStats()
        return st

    # -------------------------------------------------------------------- record
    def record(self, step=None, **scalars):
        """Fold scalars into the accumulators. O(1) per key, no history kept."""
        if not self.enabled:
            return
        for key, value in scalars.items():
            if value is None:
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            self._stats(self._interval, key).update(x)
            life = self._stats(self._lifetime, key)

            if not math.isfinite(x):
                self.event(step, "nonfinite", f"{key} is not finite", key=key, value=str(value))
            elif (key.startswith("loss/") and life.n >= self.spike_min_n
                  and life.std > 0
                  and abs(x - life.mean) > self.spike_sigma * life.std):
                self.event(step, "loss_spike",
                           f"{key} is {abs(x - life.mean) / life.std:.1f} sigma "
                           f"from its running mean",
                           key=key, value=round(x, 6),
                           running_mean=round(life.mean, 6), running_std=round(life.std, 6))
            life.update(x)

    def record_dict(self, step, mapping, prefix=""):
        if mapping:
            self.record(step, **{f"{prefix}{k}": v for k, v in mapping.items()})

    # ------------------------------------------------------------- module health
    def probe_modules(self, step, groups, lr_by_group=None, track_delta=True):
        """Gradient norm, parameter norm, their ratio, and measured movement.

        Args:
            groups: {name: nn.Module or iterable of parameters}
            lr_by_group: optional {name: lr} to log alongside.
        """
        if not self.enabled:
            return
        import torch                                    # local: keep torch optional

        for name, mod in groups.items():
            if mod is None:
                continue
            params = list(mod.parameters()) if hasattr(mod, "parameters") else list(mod)
            trainable = [p for p in params if p.requires_grad]
            g2 = 0.0
            w2 = 0.0
            n_train = 0
            with torch.no_grad():
                for p in trainable:
                    n_train += p.numel()
                    w2 += float(p.detach().float().pow(2).sum().item())
                    if p.grad is not None:
                        g2 += float(p.grad.detach().float().pow(2).sum().item())
            gnorm, wnorm = math.sqrt(g2), math.sqrt(w2)
            self.record(step, **{
                f"grad/{name}/norm": gnorm,
                f"grad/{name}/param_norm": wnorm,
                # gradient-to-weight ratio: the per-layer effective learning rate
                # driver, so this is the number that says whether the lr is sane
                f"grad/{name}/ratio": gnorm / wnorm if wnorm > 0 else float("nan"),
                f"grad/{name}/trainable": n_train,
            })
            if lr_by_group and name in lr_by_group and lr_by_group[name] is not None:
                self.record(step, **{f"lr/{name}": lr_by_group[name]})
            if track_delta and hasattr(mod, "named_parameters"):
                d, tname = self._delta.probe(name, mod)
                if d is not None:
                    self.record(step, **{f"delta/{name}": d})
                    if tname and f"delta/{name}" not in getattr(self, "_delta_named", ()):
                        self._delta_named = getattr(self, "_delta_named", set()) | {f"delta/{name}"}
                        self.event(step, "delta_tracked",
                                   f"tracking movement of {name}:{tname}",
                                   group=name, tensor=tname)

    # ---------------------------------------------------------------- latent health
    def probe_latents(self, step, diag):
        """Record collapse diagnostics and raise events on threshold crossings."""
        if not self.enabled or not diag:
            return
        self.record(step, **diag)
        for key, (op, thresh) in self.collapse_thresholds.items():
            if key not in diag:
                continue
            v = diag[key]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            crossed = v < thresh if op == "<" else v > thresh
            if crossed:
                self.event(step, "collapse_warning",
                           f"{key}={v:.5g} {op} {thresh:g}",
                           key=key, value=round(v, 6), threshold=thresh)

    # --------------------------------------------------------------------- events
    def event(self, step, kind, message, **data):
        """Anomalies go straight to disk AND into a capped in-memory ring."""
        self._n_events += 1
        rec = {"type": "event", "step": step, "kind": kind, "msg": message,
               "i": self._n_events}
        rec.update(data)
        self.events.append(rec)
        self._write(rec)

    # ---------------------------------------------------------------------- flush
    def maybe_flush(self, step, force=False):
        if not self.enabled:
            return False
        if not force and (step - self._last_flush_step) < self.log_every:
            return False
        return self.flush(step)

    def flush(self, step, extra=None):
        """Emit one interval digest and reset the interval accumulators."""
        if not self.enabled or not self._interval:
            return False
        now = time.time()
        span = max(1, step - self._last_flush_step)
        dt = max(1e-9, now - self._last_flush_time)
        rec = {
            "type": "interval", "i": self._n_intervals, "step": step,
            "steps_in_window": span,
            "it_per_s": round(span / dt, 4),
            "elapsed_h": round((now - self._t0) / 3600.0, 4),
            "metrics": {k: v.as_dict() for k, v in sorted(self._interval.items())},
        }
        if extra:
            rec.update(extra)
        self._write(rec)
        self.recent_intervals.append({"step": step, "i": self._n_intervals})
        self._n_intervals += 1
        for st in self._interval.values():
            st.reset()
        self._interval.clear()          # keys are re-created on demand: O(1) steady state
        self._last_flush_step = step
        self._last_flush_time = now
        return True

    # ---------------------------------------------------------------------- close
    def close(self, step=None, status="completed", **extra):
        if not self.enabled or self._closed:
            return
        if step is not None:
            self.flush(step)
        by_kind = {}
        for e in self.events:
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        self._write({
            "type": "summary", "status": status, "step": step,
            "intervals": self._n_intervals, "events_total": self._n_events,
            "events_retained_by_kind": by_kind,
            "elapsed_h": round((time.time() - self._t0) / 3600.0, 4),
            "delta_elements_tracked": self._delta.elements_tracked,
            "lifetime": {k: v.as_dict() for k, v in sorted(self._lifetime.items())},
            **extra,
        })
        self._closed = True

    # ------------------------------------------------------------------ accounting
    def memory_report(self):
        """Explicit accounting, so 'bounded' is checkable rather than claimed."""
        return {
            "metric_keys_interval": len(self._interval),
            "metric_keys_lifetime": len(self._lifetime),
            "floats_per_key": 7,
            "events_retained": len(self.events),
            "events_cap": self.events.maxlen,
            "intervals_retained": len(self.recent_intervals),
            "intervals_cap": self.recent_intervals.maxlen,
            "delta_elements_tracked": self._delta.elements_tracked,
            "delta_element_budget": self._delta.element_budget,
            "grows_with_steps": False,
        }
