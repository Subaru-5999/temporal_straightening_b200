#!/usr/bin/env python3
"""Turn a training_log JSONL into a compact, bounded digest of a whole run.

The point: a 12-hour run must reduce to something a reader can absorb in one
pass, whose length does NOT scale with the run length. The phase table is
downsampled to a fixed number of rows no matter how many intervals exist, so the
output is roughly the same size for 2k steps or 2M.

Usage
    python summarize_training_log.py logs/telemetry/<run>.jsonl
    python summarize_training_log.py <file> --rows 30          # phase table rows
    python summarize_training_log.py <file> --full-events      # do not cap events
    python summarize_training_log.py <file> --metrics loss,latent   # sections
    python summarize_training_log.py <file> --csv out.csv       # per-interval CSV

Paste the stdout straight into a chat: it is designed to be self-describing.
"""

import argparse
import json
import math
import os
import sys

SECTIONS = ("loss", "latent", "grad", "delta", "lr")


def load(path):
    header, intervals, events, summary = None, [], [], None
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1               # a run killed mid-write leaves a partial line
                continue
            t = rec.get("type")
            if t == "header":
                header = rec
            elif t == "interval":
                intervals.append(rec)
            elif t == "event":
                events.append(rec)
            elif t == "summary":
                summary = rec
    return header, intervals, events, summary, bad


def downsample(items, n):
    """Evenly spaced subset that always keeps the first and last item."""
    if len(items) <= n:
        return items
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def get(rec, key, field="mean"):
    m = rec.get("metrics", {}).get(key)
    if not m or m.get("n", 0) == 0:
        return None
    return m.get(field)


def fmt(v, width=9, prec=4):
    if v is None:
        return " " * (width - 1) + "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "      nan"
    if isinstance(v, float) and v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:>{width}.2e}"
    return f"{v:>{width}.{prec}f}"


def keys_in(intervals, section):
    seen = set()
    for rec in intervals:
        for k, m in rec.get("metrics", {}).items():
            if k.startswith(section + "/") and m.get("n", 0):
                seen.add(k)
    return sorted(seen)


def table(intervals, keys, rows, title):
    if not keys:
        return
    print(f"\n## {title}")
    short = [k.split("/", 1)[1] for k in keys]
    width = max(9, max((len(s) for s in short), default=9) + 1)
    print("  " + "step".rjust(9) + "".join(s.rjust(width) for s in short))
    for rec in downsample(intervals, rows):
        line = "  " + f"{rec['step']:>9,}"
        for k in keys:
            line += fmt(get(rec, k), width=width)
        print(line)


def trend(intervals, key):
    """(first, last, min, max) of a metric's interval means across the run."""
    vals = [(r["step"], get(r, key)) for r in intervals]
    vals = [(s, v) for s, v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    only = [v for _, v in vals]
    return only[0], only[-1], min(only), max(only)


def verdict(intervals):
    """The one question that matters: did the representation survive?"""
    print("\n## Verdict: did the representation survive?")
    checks = [
        ("latent/probe_r2", "linear-probe R^2 (state readable from latent)", ">", 0.2),
        ("latent/eff_rank_frac", "effective rank / dim", ">", 0.2),
        ("latent/std", "latent std across (batch, time)", ">", 1e-3),
    ]
    any_found = False
    for key, label, op, thresh in checks:
        tr = trend(intervals, key)
        if tr is None:
            continue
        any_found = True
        first, last, lo, hi = tr
        ok = last > thresh if op == ">" else last < thresh
        arrow = "->"
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
        print(f"          {first:.4g} {arrow} {last:.4g}   (min {lo:.4g}, max {hi:.4g}, "
              f"needs {op} {thresh:g})")
    if not any_found:
        print("  no latent diagnostics in this log (training.log_diagnostics=False?)")
        return
    tr_loss = trend(intervals, "loss/z_visual_loss") or trend(intervals, "loss/loss")
    tr_r2 = trend(intervals, "latent/probe_r2")
    if tr_loss and tr_r2:
        loss_fell = tr_loss[1] < 0.5 * tr_loss[0]
        r2_fell = tr_r2[1] < 0.5 * max(tr_r2[0], 1e-9)
        if loss_fell and r2_fell:
            print("\n  !! COLLAPSE SIGNATURE: the prediction loss fell while probe R^2")
            print("     fell with it. The loss went down by making the representation")
            print("     uninformative. This run is not usable.")
        elif loss_fell and not r2_fell:
            print("\n  Prediction loss fell while probe R^2 held: this is what a")
            print("  healthy run looks like.")


def encoder_movement(intervals):
    keys = keys_in(intervals, "delta")
    if not keys:
        return
    print("\n## Encoder movement (measured L2 change per probe interval)")
    for k in keys:
        tr = trend(intervals, k)
        if tr is None:
            continue
        first, last, lo, hi = tr
        name = k.split("/", 1)[1]
        moved = hi > 0
        note = "" if moved else "   <-- NEVER MOVED"
        print(f"  {name:<28} first {first:.3e}  last {last:.3e}  max {hi:.3e}{note}")


def event_digest(events, cap):
    if not events:
        print("\n## Events\n  none")
        return
    by_kind = {}
    for e in events:
        by_kind.setdefault(e.get("kind", "?"), []).append(e)
    print(f"\n## Events ({len(events)} retained)")
    for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  {kind}  x{len(group)}")
        show = group if cap is None else group[:cap]
        for e in show:
            print(f"    step {str(e.get('step')):>9}  {e.get('msg','')}")
        if cap is not None and len(group) > cap:
            print(f"    ... {len(group) - cap} more (use --full-events)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--rows", type=int, default=22,
                    help="phase-table rows; output size is independent of run length")
    ap.add_argument("--metrics", default=",".join(SECTIONS),
                    help=f"comma-separated sections to show, from {SECTIONS}")
    ap.add_argument("--events", type=int, default=6, help="events shown per kind")
    ap.add_argument("--full-events", action="store_true")
    ap.add_argument("--csv", help="also write every interval mean to CSV")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"no such file: {args.path}")
    header, intervals, events, summary, bad = load(args.path)

    print("=" * 78)
    print(f"TRAINING TELEMETRY  {header.get('run','?') if header else '?'}")
    print("=" * 78)
    if header:
        print(f"  started      : {header.get('started')}")
        print(f"  log_every    : {header.get('log_every')} steps")
        cfg = header.get("config") or {}
        if cfg:
            print("  config       :")
            for k in sorted(cfg):
                print(f"      {k} = {cfg[k]}")
    print(f"  intervals    : {len(intervals)}")
    if intervals:
        print(f"  steps        : {intervals[0]['step']:,} -> {intervals[-1]['step']:,}")
        print(f"  elapsed      : {intervals[-1].get('elapsed_h', 0):.2f} h "
              f"(last window {intervals[-1].get('it_per_s', 0):.2f} it/s)")
    if bad:
        print(f"  NOTE         : {bad} truncated line(s) -- run was killed mid-write")
    if summary:
        print(f"  status       : {summary.get('status')} at step {summary.get('step')}")
        if summary.get("events_retained_by_kind"):
            print(f"  events       : {summary['events_retained_by_kind']}")
    elif intervals:
        print("  status       : NO SUMMARY RECORD -- run died or is still going")

    if not intervals:
        sys.exit("\nno interval records; nothing to summarise")

    wanted = [s.strip() for s in args.metrics.split(",") if s.strip()]
    titles = {
        "loss": "Loss terms (interval means)",
        "latent": "Latent health (interval means)",
        "grad": "Gradient / weight norms and their ratio",
        "delta": "Weight movement",
        "lr": "Learning rates",
    }
    for section in wanted:
        if section == "delta":
            continue
        table(intervals, keys_in(intervals, section), args.rows,
              titles.get(section, section))

    if "delta" in wanted:
        encoder_movement(intervals)
    verdict(intervals)
    event_digest(events, None if args.full_events else args.events)

    if args.csv:
        import csv
        all_keys = sorted({k for r in intervals for k in r.get("metrics", {})})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step"] + all_keys)
            for r in intervals:
                w.writerow([r["step"]] + [get(r, k) for k in all_keys])
        print(f"\nwrote {args.csv} ({len(intervals)} rows x {len(all_keys)} metrics)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
