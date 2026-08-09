#!/usr/bin/env python3
"""collect_final_results.py -- merge a run's final-protocol artifacts into one
RESULTS.md so nothing has to be read back out of logs by hand.

Inputs (all optional; a missing file is reported as "not produced"):
    results/<name>.json                     success rates + timing (reproduce_table1)
    <run>/final_eval/rollout_drift.json     snr / beat / drift per horizon
    <run>/final_eval/planning_jacobian.json condition-number proxy, cf straightness
    <run>/final_eval/landscape_<name>.json  planning-cost slice scalars
    <run>/final_eval/metric_alignment.json  probes / Spearman / nn-state ratio
    <run>/hydra.yaml                        the objective that was actually trained

Output:
    <run>/final_eval/RESULTS.md  (also printed)

Usage:  python collect_final_results.py checkpoints/test/<run>
"""

import json
import os
import sys

from omegaconf import OmegaConf

# Recorded anchors for the PushT H=5 protocol (PROGRESS_SIGREG_E2E.md): the
# bar any "better success rate" claim must clear, and where it stood before.
ANCHORS = [
    ("paper frozen DINOv2 baseline", "77.33 +/- 6.18", "85.33 +/- 4.99"),
    ("e2e + SIGReg + straightening", "13.33 +/- 1.15", "44.67 +/- 10.26"),
    ("e2e + SIGReg + straightening + grounding", "30.00 +/- 7.21", "40.00 +/- 10.39"),
]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def first_entry(payload, name):
    """These experiment JSONs are keyed by run basename; be tolerant."""
    if not isinstance(payload, dict):
        return None
    if name in payload:
        return payload[name]
    if len(payload) == 1:
        return next(iter(payload.values()))
    return None


def cell(rec, key):
    if not rec or key not in rec or rec[key] is None:
        return "—"
    d = rec[key]
    if d.get("mean") is None:
        return "—"
    seeds = ", ".join(str(s) for s in d.get("seeds", []))
    return f"{d['mean']:.2f} +/- {d.get('std', float('nan')):.2f}  (n={d.get('n')}; {seeds})"


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = os.path.abspath(sys.argv[1].rstrip("/\\"))
    name = os.path.basename(run)
    fe = os.path.join(run, "final_eval")
    lines = []
    w = lines.append

    w(f"# Final protocol results -- `{name}`")
    w("")

    cfg = None
    try:
        cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))
    except Exception as e:
        w(f"(hydra.yaml unreadable: {e})")
    if cfg is not None:
        tr = cfg.training
        w("## Objective (as trained)")
        w("")
        w("| setting | value |")
        w("|---|---|")
        w(f"| straighten | {tr.get('straighten')} (curv_on={tr.get('curv_on')}) |")
        w(f"| sigreg | {tr.get('sigreg')}, coeff={tr.get('sigreg_coeff')}, "
          f"proj={tr.get('sigreg_num_proj')}, apply_to={tr.get('sigreg_apply_to')} |")
        w(f"| ground_proprio | {tr.get('ground_proprio')}, "
          f"dims={tr.get('ground_proprio_dims')} |")
        w(f"| cf_curv | {tr.get('cf_curv')}, H={tr.get('cf_H')}, "
          f"mode={tr.get('cf_mode')} |")
        w(f"| act_sens | {tr.get('act_sens')}, margin={tr.get('act_sens_margin')} |")
        w(f"| freeze_backbone / stop_grad | {tr.get('freeze_backbone')} / "
          f"{tr.get('stop_grad')} |")
        w(f"| lr (encoder / backbone) | {tr.get('encoder_lr')} / "
          f"{tr.get('backbone_lr')} |")
        w(f"| budget | max_iterations={tr.get('max_iterations')}, "
          f"epochs={tr.get('epochs')} |")
        w("")

    succ = load_json(os.path.join("results", f"{name}.json"))
    w("## Success rates (paper protocol: 50 samples, goal_H=25, seeds 100/200/300)")
    w("")
    w("| arm | this run |")
    w("|---|---|")
    w(f"| open-loop GD | {cell(succ, 'open_loop')} |")
    w(f"| MPC (GD) | {cell(succ, 'mpc')} |")
    w(f"| open-loop CEM | {cell(succ, 'cem_open_loop')} |")
    w("")
    w("Anchors (PushT, H=5):")
    w("")
    w("| configuration | open-loop | MPC |")
    w("|---|---|---|")
    for label, ol, mpc in ANCHORS:
        w(f"| {label} | {ol} | {mpc} |")
    w("")
    if succ and succ.get("timing"):
        w("Timing: " + json.dumps(succ["timing"]))
        w("")

    rd = first_entry(load_json(os.path.join(fe, "rollout_drift.json")), name)
    w("## Rollout quality (rollout_drift.py)")
    w("")
    if rd:
        h = min(rd["horizon"], rd["K"]) - 1
        w("| channel | snr@K | beat@K | drift@K |")
        w("|---|---|---|---|")
        for ch in ("visual", "proprio"):
            m = rd[ch]
            w(f"| {ch} | {m['snr'][h]:.2f} | {m['beat'][h]:.3f} | "
              f"{m['drift'][h]:.3f} |")
        w("")
        w(f"(horizon K={rd['K']} model steps, {rd['n_windows']} windows)")
    else:
        w("not produced")
    w("")

    jac = first_entry(load_json(os.path.join(fe, "planning_jacobian.json")), name)
    w("## Planning Jacobian -- condition-number proxy (planning_jacobian.py)")
    w("")
    if jac:
        j = jac["jacobian"]
        w(f"- cond_proxy (sigma_max/sigma_min): **{j['cond']:.3g}**")
        w(f"- eff_rank: {j['eff_rank']:.2f}, active dims: {j['active_dims']}, "
          f"top-1 energy {j['top1_energy'] * 100:.1f}%")
        w(f"- counterfactual straightness: true actions "
          f"{jac['cf_cos_true']:+.3f} vs wrong actions {jac['cf_cos_alt']:+.3f}")
    else:
        w("not produced")
    w("")

    land = load_json(os.path.join(fe, f"landscape_{name}.json"))
    w("## Planning-cost landscape (planning_landscape.py)")
    w("")
    if land:
        s = land["scalars"]
        w(f"- J(zero)={s['J_zero']:.4g}, J(a*)={s['J_star']:.4g}, "
          f"J(gt)={s['J_gt']:.4g}, grid min={s['J_min']:.4g}")
        w(f"- curvature toward gt {s['curv_d1']:.4g} vs orthogonal "
          f"{s['curv_d2']:.4g} -> anisotropy **{land['anisotropy']:.3g}**")
    else:
        w("not produced")
    w("")

    ma = load_json(os.path.join(fe, "metric_alignment.json"))
    w("## Metric alignment (metric_alignment.py)")
    w("")
    if ma:
        flat = flatten(ma)
        shown = [(k, v) for k, v in sorted(flat.items())
                 if not k.startswith(("pairs", "frames"))]
        for k, v in shown[:40]:
            w(f"- `{k}`: {v}")
    else:
        w("not produced")
    w("")

    os.makedirs(fe, exist_ok=True)
    out = os.path.join(fe, "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
