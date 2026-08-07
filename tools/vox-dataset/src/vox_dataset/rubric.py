"""The rubric engine: score a dataset report against a per-target-model profile.

A profile (``profiles/<model>.yaml``) is a list of checks. Each check names an aggregate metric,
an operator, a target, a weight and a severity, plus a remediation template. The engine evaluates
every check to pass / warn / fail / na, and combines the weighted credits into a 0-100 score:

    credit  pass = 1.0   warn = 0.5   fail = 0.0
    score   = 100 * sum(weight * credit) / sum(weight over non-na checks)

A check is ``na`` (excluded from the score, but reported) when its metric is unavailable — e.g.
phone coverage with no transcripts. Any failed ``critical`` check sets ``critical_fail`` so a
formally-passing score can't hide a disqualifying gap.

Operators
---------
``ge le gt lt eq``   scalar ``metrics[metric]`` vs numeric ``target`` (``warn`` gives a soft zone)
``in_range``         scalar vs ``target: [lo, hi]``
``frac_in_range``    list ``lists[metric]``: fraction within ``target: [lo, hi]`` must be >= ``min``
``frac_ge``          list: fraction of values >= ``thresh`` must be >= ``min``
``frac_le``          list: fraction of values <= ``thresh`` must be >= ``min``
``dist_frac``        dict ``format[metric]`` (or metrics): ``key`` count / total must be >= ``min``

Profiles are model-EXTENSIBLE: drop a new yaml in ``profiles/`` and it is selectable by stem.
"""

from __future__ import annotations

from pathlib import Path

_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "na"
_CREDIT = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}


# --------------------------------------------------------------------------- profile loading
def profile_path(name: str) -> Path:
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = _PROFILE_DIR / f"{name}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"no rubric profile '{name}' (looked in {_PROFILE_DIR})")


def list_profiles() -> list[str]:
    return sorted(p.stem for p in _PROFILE_DIR.glob("*.yaml"))


def load_profile(name: str) -> dict:
    import yaml

    with open(profile_path(name), encoding="utf-8") as fh:
        prof = yaml.safe_load(fh)
    prof.setdefault("name", Path(name).stem)
    prof.setdefault("checks", [])
    return prof


# --------------------------------------------------------------------------- metric access
def _scalar(report: dict, metric: str):
    """Fetch a flat scalar metric from the report's ``metrics`` block (None if missing)."""
    return report.get("metrics", {}).get(metric)


def _list(report: dict, metric: str):
    return report.get("lists", {}).get(metric)


def _dist_dict(report: dict, metric: str):
    """A distribution dict for dist_frac: search format/metrics blocks and pitch/phones."""
    for block in ("format", "metrics"):
        d = report.get(block, {}).get(metric)
        if isinstance(d, dict):
            return d
    return None


# --------------------------------------------------------------------------- operators
def _cmp(op: str, v: float, target: float) -> bool:
    return {
        "ge": v >= target, "le": v <= target, "gt": v > target,
        "lt": v < target, "eq": v == target,
    }[op]


def _eval_check(check: dict, report: dict) -> dict:
    op = check["op"]
    metric = check.get("metric")
    weight = float(check.get("weight", 1.0))
    result = {
        "id": check.get("id", metric),
        "label": check.get("label", check.get("id", metric)),
        "metric": metric,
        "op": op,
        "target": check.get("target"),
        "weight": weight,
        "severity": check.get("severity", "major"),
        "status": NA,
        "value": None,
        "detail": "",
    }

    ctx: dict = {"target": check.get("target")}

    if op in ("ge", "le", "gt", "lt", "eq", "in_range"):
        v = _scalar(report, metric)
        result["value"] = v
        ctx["value"] = v
        if v is None:
            result["status"] = NA
            result["detail"] = "metric unavailable"
        elif op == "in_range":
            lo, hi = check["target"]
            ctx["lo"], ctx["hi"] = lo, hi
            result["status"] = PASS if (lo <= v <= hi) else FAIL
        else:
            target = float(check["target"])
            warn = check.get("warn")
            if _cmp(op, v, target):
                result["status"] = PASS
            elif warn is not None and _cmp(op, v, float(warn)):
                result["status"] = WARN
            else:
                result["status"] = FAIL
            if op in ("ge", "gt"):
                ctx["deficit"] = round(target - v, 3)
            elif op in ("le", "lt"):
                ctx["excess"] = round(v - target, 3)

    elif op in ("frac_in_range", "frac_ge", "frac_le"):
        lst = _list(report, metric)
        min_frac = float(check.get("min", 0.9))
        ctx["min"] = min_frac
        if not lst:
            result["status"] = NA
            result["detail"] = "no per-clip values"
        else:
            vals = [x for x in lst if x is not None]
            if op == "frac_in_range":
                lo, hi = check["target"]
                ctx["lo"], ctx["hi"] = lo, hi
                hits = sum(1 for x in vals if lo <= x <= hi)
            else:
                thresh = float(check["thresh"])
                ctx["thresh"] = thresh
                if op == "frac_ge":
                    hits = sum(1 for x in vals if x >= thresh)
                else:
                    hits = sum(1 for x in vals if x <= thresh)
            frac = hits / len(vals) if vals else 0.0
            result["value"] = round(frac, 4)
            ctx["value"] = result["value"]
            ctx["n_out"] = len(vals) - hits
            warn_frac = check.get("warn")
            if frac >= min_frac:
                result["status"] = PASS
            elif warn_frac is not None and frac >= float(warn_frac):
                result["status"] = WARN
            else:
                result["status"] = FAIL

    elif op == "dist_frac":
        d = _dist_dict(report, metric)
        min_frac = float(check.get("min", 1.0))
        key = str(check["key"])
        ctx["min"] = min_frac
        ctx["key"] = key
        if not d:
            result["status"] = NA
            result["detail"] = "distribution unavailable"
        else:
            total = sum(int(x) for x in d.values())
            got = int(d.get(key, 0))
            frac = got / total if total else 0.0
            result["value"] = round(frac, 4)
            ctx["value"] = result["value"]
            warn_frac = check.get("warn")
            if frac >= min_frac:
                result["status"] = PASS
            elif warn_frac is not None and frac >= float(warn_frac):
                result["status"] = WARN
            else:
                result["status"] = FAIL
    else:
        result["status"] = NA
        result["detail"] = f"unknown op '{op}'"

    # Remediation: only shown when not a clean pass. Supports {missing} for phone-coverage checks.
    if result["status"] in (WARN, FAIL) and check.get("remediation"):
        if metric and metric.startswith("phone_coverage"):
            cov = report.get("phones", {}).get("coverage") or {}
            missing = cov.get("missing", [])
            ctx["missing"] = ", ".join(missing[:12]) + ("…" if len(missing) > 12 else "")
            ctx["n_missing"] = len(missing)
        result["remediation"] = _safe_format(check["remediation"], ctx)
    return result


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _safe_format(template: str, ctx: dict) -> str:
    return template.format_map(_SafeDict(ctx))


# --------------------------------------------------------------------------- scoring
def score(report: dict, profile: str | dict) -> dict:
    """Evaluate ``report`` against ``profile`` (name or loaded dict). Returns the scored rubric."""
    prof = profile if isinstance(profile, dict) else load_profile(profile)
    checks = [_eval_check(c, report) for c in prof.get("checks", [])]

    scored = [c for c in checks if c["status"] != NA]
    total_w = sum(c["weight"] for c in scored)
    got = sum(c["weight"] * _CREDIT[c["status"]] for c in scored)
    pct = round(100.0 * got / total_w, 1) if total_w > 0 else 0.0

    critical_fail = any(c["status"] == FAIL and c["severity"] == "critical" for c in checks)
    counts = {s: sum(1 for c in checks if c["status"] == s) for s in (PASS, WARN, FAIL, NA)}

    return {
        "profile": prof.get("name"),
        "profile_description": prof.get("description"),
        "score": pct,
        "grade": _grade(pct, critical_fail),
        "critical_fail": critical_fail,
        "counts": counts,
        "checks": checks,
    }


def _grade(pct: float, critical_fail: bool) -> str:
    if critical_fail:
        return "BLOCKED"
    if pct >= 90:
        return "READY"
    if pct >= 75:
        return "USABLE"
    if pct >= 50:
        return "MARGINAL"
    return "UNFIT"
