"""Run-directory naming for the end-to-end / SIGReg variants.

Kept stdlib-only (no hydra/omegaconf) so it can be unit-tested without the
training dependencies installed. Registered as an OmegaConf resolver by
`custom_resolvers.py` and used from the hydra run/sweep dir templates in
`conf/train.yaml`.

Contract: for the original defaults the tag is the empty string, so every
pre-existing checkpoint path stays byte-identical and the Table-1 reproductions
remain valid and diffable. Only opt-in variants get a suffix.
"""


def truthy(value) -> bool:
    """OmegaConf may hand over a real bool or the string 'False'/'None'."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "false", "none", "null", "0")


def fmt_coeff(value) -> str:
    """Render a coefficient the way the run names already do: 0.1 -> '1e-1'."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == 0:
        return "0"
    mantissa, exp = f"{f:.0e}".split("e")
    return f"{mantissa}e{int(exp)}"


def variant_tag(sigreg, sigreg_coeff, freeze_backbone, curv_on) -> str:
    """Suffix describing the objective/trainability variant.

    Examples:
        (False, 0.0, True,  'features') -> ''                 (baseline, unchanged)
        (True,  0.1, False, 'features') -> '_sig1e-1_e2e'
        (True,  0.1, False, 'velocity') -> '_sig1e-1_e2e_curvvel'
        (False, 0.0, False, 'features') -> '_e2e'             (negative control)
    """
    parts = []
    if truthy(sigreg):
        try:
            coeff = float(sigreg_coeff or 0)
        except (TypeError, ValueError):
            coeff = 0.0
        if coeff > 0:
            parts.append(f"sig{fmt_coeff(sigreg_coeff)}")
    if not truthy(freeze_backbone):
        parts.append("e2e")
    if str(curv_on) == "velocity":
        parts.append("curvvel")
    return ("_" + "_".join(parts)) if parts else ""
