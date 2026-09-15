#!/usr/bin/env python3
"""
multimorbidity.py — SRDC-facing multimorbidity post-processor.

Stateless. Takes the per-disease calibrated probabilities that
StagePredictor.predict_row already returns and derives the joint
quantities. No models, no retraining, no new artifacts.

Design decisions worth knowing before integrating
-------------------------------------------------
1. RANK ONLY. Singleton E/O runs 2-3x for hospital-ascertained
   conditions under the England-train / Scotland+Wales-test split
   (Deviation 7; Clifton et al. 2026). A pair probability multiplies
   two of those, so P(both) is off by 4-9x and a triplet by 8-27x.
   Absolute multimorbidity risk is therefore LESS deployable than
   absolute singleton risk, not more. Every output here carries
   risk_output_validated = "rank_only". Do not render percentages.

2. EVIDENCE TIERS ON PAIRS. A pair probability is reported as
   "validated" only if that specific pair cleared the >= 30 co-event
   floor in the D6.2 test-set evaluation (pair_evaluation.csv) for the
   patient's stratum. Everything else is "unvalidated" — the arithmetic
   is identical but no empirical discrimination was measurable, so the
   consumer must not present it with the same confidence. Coverage is
   uneven by design: 28 evaluable pairs in Male 60+, 2 in Female <60.

3. NO CONDITION IS WITHHELD BY DEFAULT. Earlier versions withheld four
   cancers on the grounds that national screening programmes govern
   their detection. That reasoning rested on a UK screening context,
   and STAGE deploys across health systems whose programmes differ, so
   the judgement did not belong hardcoded in this file. The empirical
   equivalent ships instead: pair_evaluation.csv carries an age-only
   baseline auc per pair and a beats_age_baseline flag, which applies
   in any health system. in male 60+ the two pairs that fail it are
   both cancer prostate pairs, so the signal survives the change.

4. CAUSAL SEQUENCE IS NOT CO-OCCURRENCE. Stroke -> Dementia and
   AF -> Heart Failure are cascades, not simultaneous events, and a
   10-year binary window collapses that distinction. Pairs flagged in
   CAUSAL_SEQUENCE_PAIRS carry a note so a clinical UI can phrase them
   as progression rather than co-morbidity.

Usage
-----
    from multimorbidity import (
        load_reference, multimorbidity_from_singletons,
    )

    ref = load_reference(
        model_card_path="model_card_pipeline.json",
        pair_eval_dir="multimorbidity/",
    )

    probs = predictor.predict_row(persona)          # {disease: prob}
    mm = multimorbidity_from_singletons(
        probs, stratum="Male_60plus_srdc_poc", reference=ref,
    )
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

RISK_OUTPUT_VALIDATED = "rank_only"

# Pairs where the literature supports a directional mechanism rather than
# mere co-occurrence. Not exhaustive; extend after clinical review.
CAUSAL_SEQUENCE_PAIRS: dict[frozenset, str] = {
    frozenset(
        {"Stroke Cerebrovascular", "Dementia (ICD10)"}
    ): "Post-stroke cognitive decline — typically sequential, not concurrent.",
    frozenset(
        {"Atrial Fibrillation", "Heart Failure"}
    ): "AF can precipitate HF via chronic tachycardia; bidirectional.",
    frozenset(
        {"Diabetes Mellitus", "Chronic Kidney Disease"}
    ): "Diabetic nephropathy — CKD typically follows diabetes.",
    frozenset(
        {"Hypertension", "Chronic Kidney Disease"}
    ): "Hypertensive nephropathy — usually sequential.",
    frozenset(
        {"Coronary Heart Disease", "Heart Failure"}
    ): "Ischaemic cardiomyopathy — HF commonly follows CHD.",
}


# No condition is withheld here. The card's display_policy field was withdrawn
# pending clinical review for the same reason: the screening-programme
# reasoning is UK-specific, and a deployment-site judgement does not belong in
# the file every site integrates against. A consumer that wants to withhold
# conditions should filter `probs` before calling, using its own local rule.


# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_reference(
    model_card_path: str | Path,
    pair_eval_dir: str | Path | None = None,
) -> dict:
    """Build the reference bundle: deployed disease sets + pair evidence.

    model_card_path — model_card_pipeline.json, read for the deployed
        disease set of each stratum.
    pair_eval_dir — directory holding {stratum}/pair_evaluation.csv from
        c4_multimorbidity. Optional: without it every pair is reported
        as unvalidated.
    """
    with open(model_card_path) as fh:
        card = json.load(fh)

    # The card is {_meta, strata: {stem: {per_disease: [...]}}}. Reading
    # card["per_disease"] returned an empty list on every call, which silently
    # disabled the eligibility filter — every disease entered the count.
    strata = card.get("strata")
    if not strata:
        raise ValueError(
            f"{model_card_path} has no 'strata' key. Expected "
            f"model_card_pipeline.json as produced by generate_model_card.py."
        )
    deployed: dict[str, set[str]] = {
        stem: {e["disease"] for e in c.get("per_disease", []) if e.get("disease")}
        for stem, c in strata.items()
    }

    any_stratum = next(iter(strata.values()))
    risk_output = any_stratum.get("deployment_policy", {}).get(
        "risk_output_validated", RISK_OUTPUT_VALIDATED
    )
    validated: dict[str, dict[frozenset, dict]] = {}
    if pair_eval_dir is not None:
        pair_eval_dir = Path(pair_eval_dir)
        for csv_path in pair_eval_dir.glob("*/pair_evaluation.csv"):
            stratum = csv_path.parent.name
            rows: dict[frozenset, dict] = {}
            with open(csv_path) as fh:
                header = fh.readline().strip().split(",")
                for line in fh:
                    if not line.strip():
                        continue
                    vals = _split_csv_line(line.strip(), len(header))
                    rec = dict(zip(header, vals))
                    key = frozenset({rec["disease_j"], rec["disease_k"]})
                    rows[key] = {
                        "auc": _maybe_float(rec.get("auc_joint")),
                        "auc_ci_lo": _maybe_float(rec.get("auc_ci_lo")),
                        "auc_ci_hi": _maybe_float(rec.get("auc_ci_hi")),
                        "oe_ratio": _maybe_float(rec.get("OE_ratio")),
                        "n_joint_events": _maybe_int(rec.get("n_joint_events")),
                    }
            validated[stratum] = rows

    return {
        "deployed": deployed,
        "validated_pairs": validated,
        "risk_output_validated": risk_output,
    }


def _split_csv_line(line: str, n: int) -> list[str]:
    """Split a simple CSV line. Disease names may contain parentheses but
    not commas in our naming scheme, so a plain split is safe here."""
    parts = line.split(",")
    if len(parts) == n:
        return parts
    return parts[:n] + [""] * max(0, n - len(parts))


def _maybe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CORE MATHS
# ═══════════════════════════════════════════════════════════════════════════════


def poisson_binomial_pmf(probs: list[float]) -> list[float]:
    """Exact P(N = k) for one person. DP recursion, O(D^2), D <= 26.

    Returns a list of length D+1 where index k is P(N = k).
    """
    pmf = [1.0] + [0.0] * len(probs)
    for i, p in enumerate(probs, start=1):
        for k in range(i, 0, -1):
            pmf[k] = pmf[k] * (1.0 - p) + pmf[k - 1] * p
        pmf[0] *= 1.0 - p
    return pmf


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def multimorbidity_from_singletons(
    probs: dict[str, float],
    stratum: str | None = None,
    reference: dict | None = None,
    top_k: int = 5,
    validated_pairs_only: bool = False,
) -> dict:
    """Derive multimorbidity quantities from per-disease singleton probabilities.

    probs — {disease_name: calibrated_probability} as returned by
        StagePredictor.predict_row. None / NaN values are dropped
        (patient already has the condition, so it cannot be incident).
    stratum — used to select the right validated-pair evidence table.
    reference — output of load_reference(). Required: it carries the
        deployed disease set per stratum, which is what makes the
        eligibility check meaningful.
    top_k — how many pairs to return.
    Every condition deployed in the stratum enters the count. Whether a
    given pair is worth surfacing is an empirical question answered by
    beats_age_baseline in pair_evaluation.csv, not a fixed exclusion list.
    validated_pairs_only — if True, drop pairs with no empirical evidence
        rather than returning them flagged.

    Returns a dict ready to serialise to the CDS Hooks response.
    """
    if not reference or "deployed" not in reference:
        raise ValueError(
            "reference is required. Call load_reference(model_card_path=...) "
            "first. Running without it would count every condition passed in, "
            "including conditions not deployed in this stratum."
        )
    deployed = reference["deployed"]
    if stratum not in deployed:
        raise ValueError(f"Unknown stratum {stratum!r}. Known: {sorted(deployed)}")
    deployed_here = deployed[stratum]
    validated = reference.get("validated_pairs", {}).get(stratum or "", {})

    # ── Eligibility ──────────────────────────────────────────────────────────
    eligible: dict[str, float] = {}
    excluded: dict[str, list[str]] = {}
    for disease, p in probs.items():
        if p is None or p != p:  # None or NaN — prevalent at baseline
            continue
        if disease not in deployed_here:
            excluded.setdefault("not_deployed_in_stratum", []).append(disease)
            continue
        eligible[disease] = float(p)

    names = sorted(eligible)
    values = [eligible[n] for n in names]

    if not names:
        return {
            "risk_output_validated": RISK_OUTPUT_VALIDATED,
            "n_conditions_included": 0,
            "note": "No conditions eligible.",
            "excluded": excluded,
        }

    # ── Count distribution ───────────────────────────────────────────────────
    pmf = poisson_binomial_pmf(values)
    expected_count = sum(values)
    p_ge = {k: sum(pmf[k:]) for k in (1, 2, 3, 4) if k <= len(values)}
    count_dist = {str(k): round(pmf[k], 5) for k in range(min(5, len(pmf)))}
    if len(pmf) > 5:
        count_dist["5+"] = round(sum(pmf[5:]), 5)

    # ── Pair ranking ─────────────────────────────────────────────────────────
    pair_rows = []
    for a, b in combinations(names, 2):
        key = frozenset({a, b})
        ev = validated.get(key)
        if ev is None and validated_pairs_only:
            continue
        joint = eligible[a] * eligible[b]
        row = {
            "conditions": [a, b],
            "joint_score": round(joint, 6),
            "individual_scores": [
                round(eligible[a], 6),
                round(eligible[b], 6),
            ],
            "evidence": "validated" if ev else "unvalidated",
        }
        if ev:
            row["evidence_detail"] = {
                "test_set_auc": ev["auc"],
                "auc_ci_95": [ev["auc_ci_lo"], ev["auc_ci_hi"]],
                "observed_expected_ratio": ev["oe_ratio"],
                "n_joint_events_observed": ev["n_joint_events"],
            }
        note = CAUSAL_SEQUENCE_PAIRS.get(key)
        if note:
            row["sequence_note"] = note
        pair_rows.append(row)

    pair_rows.sort(key=lambda r: r["joint_score"], reverse=True)
    top_pairs = pair_rows[:top_k]

    return {
        "risk_output_validated": RISK_OUTPUT_VALIDATED,
        "interpretation": (
            "Scores rank this patient relative to the modelled population. "
            "They are not calibrated absolute probabilities and must not be "
            "displayed as percentages."
        ),
        "n_conditions_included": len(names),
        "conditions_included": names,
        "expected_condition_count": round(expected_count, 4),
        "count_distribution": count_dist,
        "probability_at_least": {f"ge_{k}": round(v, 5) for k, v in p_ge.items()},
        "top_pairs": top_pairs,
        "n_pairs_validated": sum(1 for r in pair_rows if r["evidence"] == "validated"),
        "n_pairs_total": len(pair_rows),
        "excluded": excluded,
        "caveats": [
            "Absolute joint risk compounds singleton calibration error "
            "(2-3x for singletons becomes 4-9x for pairs). Rank only.",
            "Pair evidence coverage varies by stratum: pairs marked "
            "unvalidated had too few observed co-events (<30) in the D6.2 "
            "test set to measure discrimination.",
            "Competing mortality is not modelled; counts are risk-of-event "
            "counts, not expected disease burden among survivors.",
            "Counts are unweighted — two conditions may differ greatly in "
            "clinical severity.",
            "The count is over the conditions deployed in this stratum. "
            "Where local guidance says a condition should not be shown "
            "alongside others - for example one governed by a national "
            "screening programme - filter it out of the probabilities "
            "before calling this function.",
        ],
    }
