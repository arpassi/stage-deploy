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

2. EVIDENCE TIERS ON PAIRS AND TRIPLETS. Every pair and triplet of
   eligible conditions is scored and returned; nothing is filtered on
   evidence. evidence_tier records what the D6.2 test set
   (pair_evaluation.csv / triplet_evaluation.csv) supports for that
   specific combination in the patient's stratum:
     validated  >= 30 co-events: O, prevalence, O/E with bootstrap CI,
                AUC with bootstrap CI and the age-only comparison.
     low_n      5-29 co-events: O, prevalence and O/E with an exact
                Poisson CI. No AUC.
     suppressed < 5 co-events: no counts (UK Biobank output rule).
     not_evaluated  no row, e.g. valid cohort under 100 persons.
   evidence is "validated" for the first tier and "unvalidated" for the
   rest — the arithmetic is identical but no discrimination was
   measurable, so the consumer must not present it with the same
   confidence. Coverage is uneven by design: 28 validated pairs in
   Male 60+, 2 in Female <60; for triplets, at most CHD+HF+AF in
   Male 60+.

3. NO CONDITION IS WITHHELD BY DEFAULT. Earlier versions withheld four
   cancers on the grounds that national screening programmes govern
   their detection. That reasoning rested on a UK screening context,
   and STAGE deploys across health systems whose programmes differ, so
   the judgement did not belong hardcoded in this file. The empirical
   equivalent ships instead: each validated pair or triplet carries
   auc_gain_over_age and beats_age_baseline in evidence_detail. It is a
   flag, not a filter — a validated item that ranks worse than age alone
   is still returned, marked as such. In Male 60+ the two pairs that fail
   it are both Cancer Prostate pairs, so the signal survives the change.

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
    # Evidence tables hold every evaluated combination with its tier. The
    # reference keys keep their historical names; filter on
    # ev["evidence_tier"] == "validated" for the >= 30 co-event subset.
    validated: dict[str, dict[frozenset, dict]] = {}
    validated_triplets: dict[str, dict[frozenset, dict]] = {}
    if pair_eval_dir is not None:
        pair_eval_dir = Path(pair_eval_dir)
        for csv_path in pair_eval_dir.glob("*/pair_evaluation.csv"):
            validated[csv_path.parent.name] = _load_evidence(csv_path, 2)
        # A stratum with no triplet_evaluation.csv reports every triplet
        # as not_evaluated.
        for csv_path in pair_eval_dir.glob("*/triplet_evaluation.csv"):
            validated_triplets[csv_path.parent.name] = _load_evidence(csv_path, 3)

    return {
        "deployed": deployed,
        "validated_pairs": validated,
        "validated_triplets": validated_triplets,
        "risk_output_validated": risk_output,
    }


def _load_evidence(csv_path: Path, n_members: int) -> dict[frozenset, dict]:
    """One stratum's pair (2) or triplet (3) evidence table, keyed by the
    frozenset of disease names. Pre-tier files held only >= 30-event rows,
    so a missing evidence_tier column means validated."""
    members = ["disease_j", "disease_k", "disease_l"][:n_members]
    prev_col = "pair_prevalence" if n_members == 2 else "triplet_prevalence"
    rows: dict[frozenset, dict] = {}
    with open(csv_path) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            if not line.strip():
                continue
            rec = dict(zip(header, _split_csv_line(line.strip(), len(header))))
            key = frozenset(rec[m] for m in members)
            rows[key] = {
                "evidence_tier": rec.get("evidence_tier") or "validated",
                "auc": _maybe_float(rec.get("auc_joint")),
                "auc_ci_lo": _maybe_float(rec.get("auc_ci_lo")),
                "auc_ci_hi": _maybe_float(rec.get("auc_ci_hi")),
                "oe_ratio": _maybe_float(rec.get("OE_ratio")),
                "oe_ci_lo": _maybe_float(rec.get("OE_ci_lo")),
                "oe_ci_hi": _maybe_float(rec.get("OE_ci_hi")),
                "oe_ci_method": rec.get("OE_ci_method") or None,
                "n_joint_events": _maybe_int(rec.get("n_joint_events")),
                "prevalence": _maybe_float(rec.get(prev_col)),
                "auc_gain_over_age": _maybe_float(rec.get("auc_gain_over_age")),
                "beats_age_baseline": _maybe_bool(rec.get("beats_age_baseline")),
            }
    return rows


def _evidence_fields(ev: dict | None) -> dict:
    """evidence / evidence_tier / evidence_detail for one pair or triplet.

    validated -> full detail; low_n -> descriptive detail only (no AUC);
    suppressed and not_evaluated -> no detail.
    """
    tier = ev["evidence_tier"] if ev else "not_evaluated"
    out = {
        "evidence": "validated" if tier == "validated" else "unvalidated",
        "evidence_tier": tier,
    }
    if tier in ("validated", "low_n"):
        detail = {
            "n_joint_events_observed": ev["n_joint_events"],
            "test_set_prevalence": ev["prevalence"],
            "observed_expected_ratio": ev["oe_ratio"],
            "oe_ci_95": [ev["oe_ci_lo"], ev["oe_ci_hi"]],
            "oe_ci_method": ev["oe_ci_method"],
        }
        if tier == "validated":
            detail.update(
                test_set_auc=ev["auc"],
                auc_ci_95=[ev["auc_ci_lo"], ev["auc_ci_hi"]],
                auc_gain_over_age=ev["auc_gain_over_age"],
                beats_age_baseline=ev["beats_age_baseline"],
            )
        out["evidence_detail"] = detail
    return out


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


def _maybe_bool(v) -> bool | None:
    """'True'/'False' as pandas writes them; blank (not estimable) -> None."""
    return {"True": True, "False": False}.get(v)


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
    top_k_triplets: int = 5,
    validated_triplets_only: bool = False,
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
    validated_pairs_only — if True, drop pairs outside the validated tier
        rather than returning them flagged. Default False: shown, flagged.
    top_k_triplets — how many triplets to return.
    validated_triplets_only — as validated_pairs_only, for triplets.

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
        fields = _evidence_fields(validated.get(key))
        if validated_pairs_only and fields["evidence"] != "validated":
            continue
        joint = eligible[a] * eligible[b]
        row = {
            "conditions": [a, b],
            "joint_score": round(joint, 6),
            "individual_scores": [
                round(eligible[a], 6),
                round(eligible[b], 6),
            ],
            **fields,
        }
        note = CAUSAL_SEQUENCE_PAIRS.get(key)
        if note:
            row["sequence_note"] = note
        pair_rows.append(row)

    pair_rows.sort(key=lambda r: r["joint_score"], reverse=True)
    top_pairs = pair_rows[:top_k]

    # ── Triplet ranking (same tiers as pairs) ────────────────────────────────
    validated_tri = reference.get("validated_triplets", {}).get(stratum or "", {})
    triplet_rows = []
    for a, b, c in combinations(names, 3):
        key = frozenset({a, b, c})
        fields = _evidence_fields(validated_tri.get(key))
        if validated_triplets_only and fields["evidence"] != "validated":
            continue
        row = {
            "conditions": [a, b, c],
            "joint_score": round(eligible[a] * eligible[b] * eligible[c], 8),
            "individual_scores": [round(eligible[d], 6) for d in (a, b, c)],
            **fields,
        }
        notes = [
            CAUSAL_SEQUENCE_PAIRS[frozenset(p)]
            for p in combinations((a, b, c), 2)
            if frozenset(p) in CAUSAL_SEQUENCE_PAIRS
        ]
        if notes:
            row["sequence_notes"] = notes
        triplet_rows.append(row)

    triplet_rows.sort(key=lambda r: r["joint_score"], reverse=True)

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
        "top_triplets": triplet_rows[:top_k_triplets],
        "n_triplets_validated": sum(
            1 for r in triplet_rows if r["evidence"] == "validated"
        ),
        "n_triplets_total": len(triplet_rows),
        "excluded": excluded,
        "caveats": [
            "Absolute joint risk compounds singleton calibration error "
            "(2-3x for singletons becomes 4-9x for pairs and 8-27x for "
            "triplets). Rank only.",
            "Pair and triplet evidence coverage varies by stratum: items "
            "marked unvalidated had too few observed co-events (<30) in the "
            "D6.2 test set to measure discrimination. evidence_tier low_n "
            "(5-29 co-events) still carries the observed count, prevalence "
            "and O/E; suppressed (<5) carries no counts. Validated triplets "
            "are rare (at most CHD+HF+AF in Male 60+).",
            "beats_age_baseline = false means that item ranked patients no "
            "better than age alone in the D6.2 test set. It is shown, not "
            "hidden.",
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