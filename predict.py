#!/usr/bin/env python3
"""
predict.py — Local, artifact-only inference for the STAGE production models.

Turns raw personas (engineered descriptive-feature space, the same space the
preprocessing artifacts were fit on) into Platt-calibrated 10-year incident-risk
probabilities for every trained disease in a stratum.

Everything is driven by saved artifacts — no server paths, no dependency on the
training modules (c4_helpers / preprocessing):

  artifacts/cappers/c1_strata_{stem}_capper.joblib     {num_cols, bounds}
  artifacts/imputers/c1_strata_{stem}_imputer.joblib   {numeric_cols, cat_cols,
        numeric_imputer (fitted MICE), cat_fill_values, missingness_indicator_cols,
        dropped_high_miss_cols}
  artifacts/ohe_artifacts/c1_strata_{stem}_ohe.json    {ohe_reference_map,
        dropped_constant_dummies}
  artifacts/scalers/c1_strata_{stem}_scaler.joblib     {scale_cols, scaler}

  outputs/c4_production/feature_selection/{stem}/feature_masks.json
  outputs/c4_production/train/{stem}/models/{disease}.joblib
  outputs/c4_production/train/{stem}/calibrators/{disease}.joblib
  outputs/c4_production/sud_augmentation/{stem}/models/{Substance_Use_Disorder.joblib,
        deployment_metadata.json}
  outputs/c4_production/sud_augmentation/{stem}/calibrators/Substance_Use_Disorder.joblib

Transform chain (mirrors preprocessing steps 4–7 exactly, applied not fitted):
  capper (clip) → missingness indicators + MICE impute → OHE encode → scale.
Because feature_selection excludes tautological features *before* top-k, each
disease's feature_masks.json is already the final feature set, so per-disease
feature reconstruction is pure artifact lookup by name.

Personas may be sparse: any feature a persona omits is added as NaN and imputed,
which is exactly how a real deployment behaves with partial data.

Programmatic use (what a driver calls per row)
----------------------------------------------
    from predict import StagePredictor
    predictor = StagePredictor(stratum="Male_all_srdc_poc")   # loads artifacts once
    risks = predictor.predict_row(persona_dict)               # -> {disease: prob}
    # or batch:
    df_out = predictor.predict(personas_df)                   # -> DataFrame

CLI
---
    python predict.py --stratum Male_all_srdc_poc \
        --personas prep_deploy/test_personas.csv --out prep_deploy/predictions.csv

    # per-row routing: give the CSV a 'stratum' column instead of --stratum
    python predict.py --personas prep_deploy/test_personas.csv
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit as _logit

# Optional — only loaded when --explain is used
_shap = None

def _get_shap():
    global _shap
    if _shap is None:
        import shap
        _shap = shap
    return _shap

# ═══════════════════════════════════════════════════════════════════════════════
# §1  DEFAULT LOCAL PATHS  (overridable; no server paths)
# ═══════════════════════════════════════════════════════════════════════════════

_FALLBACK_PROJECT_ROOT = Path("/home/alvaro-passi/MEGA/STAGE/STAGE_CODE_2")
_SUD_NAME = "Substance Use Disorder"

# Deployment-scope exclusions (distinct from training-level exclusions such as
# Asthma/Epilepsy/Anxiety). These are withheld in deployment outputs by
# default, but can be re-enabled via include_excluded for research runs.
DEPLOY_EXCLUDED_DISEASES: frozenset[str] = frozenset(
    {
        "Dementia (ICD10)",
        "Parkinsons",
        "Multiple Sclerosis",
        "Sleep Disorders",
        "Severe Mental Illness",
    }
)
_CLIP_EPS = 1e-7
_ID_CANDIDATES = ("persona_id", "persona", "name", "id", "eid")
_STRATUM_COLS = ("stratum", "STRATA", "strata")
_VALID_FEATURE_SETS = ("srdc_poc", "abacus_poc")
_PERSONA_ID_RE = re.compile(r"_(?P<sex>[FfMm])_(?P<age>\d+)")


def _find_project_root() -> Path:
    """Locate the repo root via a .stage_root sentinel; else use the known path."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".stage_root").exists():
            return p
        p = p.parent
    return _FALLBACK_PROJECT_ROOT


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stratum_from_persona_id(persona_id: str, feature_set: str) -> str | None:
    """Derive stratum from persona id token, e.g. ..._F_45 -> Female_under60_{set}."""
    m = _PERSONA_ID_RE.search(str(persona_id))
    if not m:
        return None
    sex = "Female" if m.group("sex").upper() == "F" else "Male"
    band = "under60" if int(m.group("age")) < 60 else "60plus"
    return f"{sex}_{band}_{feature_set}"


# ═══════════════════════════════════════════════════════════════════════════════
# §2  PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════


class StagePredictor:
    """
    Loads all artifacts for one stratum and scores personas.

    Parameters
    ----------
    stratum         : production stem, e.g. "Male_all_srdc_poc".
    project_root    : repo root (default: auto-detected / known local path).
    artifacts_dir   : override for the preprocessing artifacts root.
    models_dir      : override for the c4_production outputs root (feature masks
                      + SUD augmentation chain).
    train_dir       : override for the TUNED train/ tree (singletons +
                      calibrators).
    exclude         : extra disease names to withhold from output.
    include_excluded: if True, ignore DEPLOY_EXCLUDED_DISEASES and score every
                      trained disease (research mode).
    """

    def __init__(
        self,
        stratum: str,
        project_root: Path | None = None,
        artifacts_dir: Path | None = None,
        models_dir: Path | None = None,
        train_dir: Path | None = None,
        exclude: set[str] | None = None,
        include_excluded: bool = False,
        verbose: bool = True,
    ) -> None:
        self.stratum = stratum
        self.verbose = verbose
        self.excluded: frozenset[str] = (
            frozenset(exclude or ())
            if include_excluded
            else DEPLOY_EXCLUDED_DISEASES | frozenset(exclude or ())
        )
        root = Path(project_root) if project_root else _find_project_root()
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir else root / "artifacts"
        )
        # tuned singletons
        self.train_dir = (
            Path(train_dir) if train_dir else root / "outputs" / "c4_production_tuned"
        )
        # feature masks + SUD augmentation chain (baseline tree only)
        self.models_dir = (
            Path(models_dir) if models_dir else root / "outputs" / "c4_production"
        )

        self._load_preprocessing_artifacts()
        self._load_feature_masks()
        self._discover_models()

        self._log(f"{self.stratum} | artifacts: {self.artifacts_dir}")
        self._log(
            f"  singletons (TUNED)   : {self.train_dir / 'train' / self.stratum}"
        )
        self._log(
            f"  feature masks        : "
            f"{self.models_dir / 'feature_selection' / self.stratum}"
        )
        if _SUD_NAME not in self.excluded:
            sud_chain = self.models_dir / "sud_augmentation" / self.stratum
            self._log(
                f"  SUD chain (TUNED XGB): {sud_chain}  "
                f"arm=augmented_depfeatures_k50"
            )

    # ── logging ──────────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{_ts()}] {msg}")

    # ── artifact loading ─────────────────────────────────────────────────────
    def _art_path(self, sub: str, suffix: str) -> Path:
        return self.artifacts_dir / sub / f"c1_strata_{self.stratum}_{suffix}"

    def _load_preprocessing_artifacts(self) -> None:
        capper_p = self._art_path("cappers", "capper.joblib")
        imputer_p = self._art_path("imputers", "imputer.joblib")
        ohe_p = self._art_path("ohe_artifacts", "ohe.json")
        scaler_p = self._art_path("scalers", "scaler.joblib")

        for p in (capper_p, imputer_p, ohe_p, scaler_p):
            if not p.exists():
                raise FileNotFoundError(f"Missing preprocessing artifact: {p}")

        self.capper = joblib.load(capper_p)
        self.imputer = joblib.load(imputer_p)
        with open(ohe_p) as fh:
            self.ohe = json.load(fh)
        self.scaler_art = joblib.load(scaler_p)
        self._log(
            f"Loaded preprocessing artifacts for {self.stratum} "
            f"(capper/imputer/ohe/scaler)"
        )

    def _load_feature_masks(self) -> None:
        mask_p = (
            self.models_dir / "feature_selection" / self.stratum / "feature_masks.json"
        )
        if not mask_p.exists():
            raise FileNotFoundError(f"Missing feature_masks.json: {mask_p}")
        with open(mask_p) as fh:
            self.feature_masks: dict[str, list[str]] = json.load(fh)
        self._log(f"Loaded feature masks for {len(self.feature_masks)} diseases")

    def _discover_models(self) -> None:
        self.train_models_dir = self.train_dir / "train" / self.stratum / "models"
        self.train_cal_dir = self.train_dir / "train" / self.stratum / "calibrators"
        if not self.train_models_dir.exists():
            raise FileNotFoundError(
                f"TUNED models required at {self.train_models_dir}; "
                f"falling back to the baseline tree is not permitted."
            )
        if not self.train_cal_dir.exists():
            raise FileNotFoundError(
                f"TUNED calibrators required at {self.train_cal_dir}; "
                f"falling back to the baseline tree is not permitted."
            )

        all_singletons = sorted(p.stem for p in self.train_models_dir.glob("*.joblib"))
        self.singleton_diseases = [d for d in all_singletons if d not in self.excluded]
        n_skipped = len(all_singletons) - len(self.singleton_diseases)

        # Augmented production SUD chain (mandatory unless SUD is excluded).
        sud_dir = self.models_dir / "sud_augmentation" / self.stratum
        self.sud_model_path = sud_dir / "models" / "Substance_Use_Disorder.joblib"
        self.sud_cal_path = sud_dir / "calibrators" / "Substance_Use_Disorder.joblib"
        sud_meta_p = sud_dir / "models" / "deployment_metadata.json"
        self.sud_feature_names: list[str] | None = None
        if _SUD_NAME not in self.excluded:
            if not self.sud_model_path.exists():
                raise FileNotFoundError(
                    f"SUD augmented model required at {self.sud_model_path}"
                )
            if not self.sud_cal_path.exists():
                raise FileNotFoundError(
                    f"SUD augmented calibrator required at {self.sud_cal_path}"
                )
            if not sud_meta_p.exists():
                raise FileNotFoundError(
                    f"SUD deployment_metadata.json required at {sud_meta_p}"
                )
            with open(sud_meta_p) as fh:
                deploy = json.load(fh)
            self.sud_feature_names = deploy.get("sud_model", {}).get("feature_names")
            if not self.sud_feature_names:
                raise ValueError(
                    f"SUD deployment_metadata.json at {sud_meta_p} is missing or "
                    f"has empty sud_model.feature_names"
                )
            production_arm = deploy.get("production_arm")
            if production_arm != "augmented_depfeatures_k50":
                raise ValueError(
                    f"SUD production_arm must be 'augmented_depfeatures_k50', "
                    f"got {production_arm!r} in {sud_meta_p}"
                )

        self._log(
            f"Discovered {len(self.singleton_diseases)} singleton models"
            + (f" ({n_skipped} deployment-excluded)" if n_skipped else "")
            + ("  + augmented SUD chain" if self.sud_feature_names is not None else "")
        )

    # ── preprocessing (apply-only, from artifacts) ───────────────────────────
    def _apply_capper(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, b in self.capper["bounds"].items():
            if col not in out.columns or b.get("cap_source") == "none":
                continue
            out[col] = out[col].clip(lower=b["cap_lo"], upper=b["cap_hi"])
        return out

    def _apply_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        art = self.imputer
        out = df.copy()

        # Missingness indicators (created from the base column's NaN status).
        for col in art.get("missingness_indicator_cols", []):
            base = (
                out[col] if col in out.columns else pd.Series(np.nan, index=out.index)
            )
            out[f"{col}_missing"] = base.isna().astype("int8")

        num_cols = art["numeric_cols"]
        cat_cols = art["cat_cols"]

        # Numeric MICE — reindex guarantees exact fitted column set + order.
        num_in = out.reindex(columns=num_cols)
        num_vals = art["numeric_imputer"].transform(num_in)
        num_df = pd.DataFrame(num_vals, columns=num_cols, index=out.index)

        # Categorical mode fill.
        cat_in = out.reindex(columns=cat_cols)
        cat_df = cat_in.fillna(art["cat_fill_values"])

        # Preserve passthrough features that are not part of imputer numeric/cat
        # sets (e.g., age_at_baseline and generated missingness indicators).
        passthrough_cols = [
            c for c in out.columns if c not in set(num_cols) and c not in set(cat_cols)
        ]
        passthrough_df = out[passthrough_cols] if passthrough_cols else None

        parts = [num_df, cat_df]
        if passthrough_df is not None:
            parts.append(passthrough_df)
        return pd.concat(parts, axis=1)

    def _apply_ohe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ref_map = self.ohe["ohe_reference_map"]
        new_cols: dict[str, pd.Series] = {}
        drop_orig: list[str] = []
        for col, spec in ref_map.items():
            if col not in out.columns:
                continue
            for cat in spec["encoded_categories"]:
                new_cols[f"{col}__{cat}"] = (out[col] == cat).astype("int8")
            drop_orig.append(col)
        out = out.drop(columns=drop_orig)
        if new_cols:
            # If personas already include one-hot columns with the same names,
            # prefer the artifact-driven reconstruction to avoid duplicates.
            overlap = [c for c in new_cols if c in out.columns]
            if overlap:
                out = out.drop(columns=overlap)
            out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
        dropped = [c for c in self.ohe.get("dropped_constant_dummies", []) if c in out]
        if dropped:
            out = out.drop(columns=dropped)
        return out

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        scale_cols = self.scaler_art["scale_cols"]
        missing = [c for c in scale_cols if c not in out.columns]
        if missing:
            raise KeyError(
                f"{self.stratum}: {len(missing)} scaler columns absent after "
                f"OHE (e.g. {missing[:5]}). Persona/artifact schema mismatch."
            )
        out[scale_cols] = self.scaler_art["scaler"].transform(out[scale_cols])
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw engineered features → model feature space (capper→impute→OHE→scale)."""
        x = df.reset_index(drop=True)
        x = self._apply_capper(x)
        x = self._apply_impute(x)
        x = self._apply_ohe(x)
        x = self._apply_scaler(x)
        return x

    # ── calibration ──────────────────────────────────────────────────────────
    @staticmethod
    def _calibrate(cal_lr, raw: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw, _CLIP_EPS, 1.0 - _CLIP_EPS)
        return cal_lr.predict_proba(_logit(clipped).reshape(-1, 1))[:, 1]

    def _score_model(
        self,
        model_path: Path,
        cal_path: Path | None,
        used_names: list[str],
        X_model: pd.DataFrame,
        disease: str,
    ) -> tuple[np.ndarray, bool]:
        clf = joblib.load(model_path)
        missing = [n for n in used_names if n not in X_model.columns]
        if missing:
            self._log(
                f"  WARN {disease}: {len(missing)} mask feature(s) absent, "
                f"zero-filled: {missing}"
            )
            for col in missing:
                X_model[col] = 0.0
        n_in = getattr(clf, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(used_names):
            raise ValueError(
                f"{disease}: model expects {n_in} features, reconstructed "
                f"{len(used_names)}."
            )
        raw = clf.predict_proba(X_model.reindex(columns=used_names).to_numpy())[:, 1]
        if cal_path is not None and cal_path.exists():
            return self._calibrate(joblib.load(cal_path), raw), True
        self._log(f"  {disease}: no calibrator — emitting RAW probability")
        return raw, False

    # ── public scoring ───────────────────────────────────────────────────────
    def predict(self, personas: pd.DataFrame, emit_raw: bool = False) -> pd.DataFrame:
        """Score a batch of personas (rows). Returns one output row per persona."""
        personas = personas.reset_index(drop=True)
        id_col = next((c for c in _ID_CANDIDATES if c in personas.columns), None)
        ids = personas[id_col].to_numpy() if id_col else np.arange(len(personas))

        X_model = self.transform(personas)

        out: dict[str, np.ndarray] = {"persona": ids}
        self.calibration_status: dict[str, bool] = {}

        for disease in self.singleton_diseases:
            if disease == _SUD_NAME:
                continue  # served by the augmented chain below
            used = self.feature_masks.get(disease)
            if used is None:
                self._log(f"  {disease}: no feature mask — skipped")
                continue
            try:
                vals, calibrated = self._score_model(
                    self.train_models_dir / f"{disease}.joblib",
                    self.train_cal_dir / f"{disease}.joblib",
                    used,
                    X_model,
                    disease,
                )
            except Exception as exc:
                self._log(f"  ERROR {disease}: {exc}")
                continue
            out[disease] = vals
            if emit_raw and not calibrated:
                out[f"{disease} (raw)"] = vals
            self.calibration_status[disease] = calibrated

        # Augmented production SUD.
        if self.sud_feature_names is not None:
            vals, calibrated = self._score_model(
                self.sud_model_path,
                self.sud_cal_path,
                self.sud_feature_names,
                X_model,
                _SUD_NAME,
            )
            out[_SUD_NAME] = vals
            self.calibration_status[_SUD_NAME] = calibrated

        result = pd.DataFrame(out)
        result.insert(1, "stratum", self.stratum)
        return result

    def predict_row(self, row: dict, emit_raw: bool = False) -> dict:
        """Score a single persona dict. Returns {disease: calibrated_probability}."""
        df = self.predict(pd.DataFrame([row]), emit_raw=emit_raw)
        rec = df.iloc[0].to_dict()
        return {k: v for k, v in rec.items() if k not in ("persona", "stratum")}

    # ── SHAP explanations ───────────────────────────────────────────────────
    def explain(
        self,
        personas: pd.DataFrame,
        out_dir: Path | None = None,
    ) -> dict[str, dict]:
        """
        Compute per-sample SHAP values for all deployed diseases.

        Returns {disease: {shap_values, expected_value, feature_names}}.
        If out_dir is given, also saves .npy + .json files per disease.
        """
        shap = _get_shap()

        personas = personas.reset_index(drop=True)
        X_model = self.transform(personas)

        results: dict[str, dict] = {}

        # Singleton diseases
        for disease in self.singleton_diseases:
            if disease == _SUD_NAME:
                continue
            used = self.feature_masks.get(disease)
            if used is None:
                continue

            model_path = self.train_models_dir / f"{disease}.joblib"
            if not model_path.exists():
                continue

            clf = joblib.load(model_path)
            missing = [n for n in used if n not in X_model.columns]
            if missing:
                self._log(
                    f"  SHAP WARN {disease}: {len(missing)} mask feature(s) absent, "
                    f"zero-filled: {missing}"
                )
                for col in missing:
                    X_model[col] = 0.0

            X_d = X_model.reindex(columns=used).to_numpy()

            try:
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_d)
                if isinstance(sv, list):
                    sv = sv[1]
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = float(ev[1]) if len(ev) > 1 else float(ev[0])
                else:
                    ev = float(ev)

                results[disease] = {
                    "shap_values": sv,
                    "expected_value": ev,
                    "feature_names": used,
                }
                self._log(f"  SHAP {disease}: {sv.shape}")
            except Exception as exc:
                self._log(f"  SHAP ERROR {disease}: {exc}")

        # Augmented SUD
        if self.sud_feature_names is not None:
            clf = joblib.load(self.sud_model_path)
            # XGBoost base_score monkey-patch
            try:
                bs = getattr(clf, "base_score", None)
                if isinstance(bs, str):
                    clf.base_score = float(bs.strip("[] "))
            except Exception:
                pass

            X_sud = X_model.reindex(columns=self.sud_feature_names).to_numpy()
            try:
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_sud)
                if isinstance(sv, list):
                    sv = sv[1]
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = float(ev[1]) if len(ev) > 1 else float(ev[0])
                else:
                    ev = float(ev)

                results[_SUD_NAME] = {
                    "shap_values": sv,
                    "expected_value": ev,
                    "feature_names": self.sud_feature_names,
                }
                self._log(f"  SHAP {_SUD_NAME}: {sv.shape}")
            except Exception as exc:
                self._log(f"  SHAP ERROR {_SUD_NAME}: {exc}")

        # Save to disk if out_dir given
        if out_dir is not None:
            shap_dir = out_dir / "shap"
            shap_dir.mkdir(parents=True, exist_ok=True)
            for disease, data in results.items():
                tag = disease.replace(" ", "_")
                np.save(str(shap_dir / f"{tag}_shap_values.npy"), data["shap_values"])
                np.save(str(shap_dir / f"{tag}_expected_value.npy"),
                        np.array(data["expected_value"]))
                with open(shap_dir / f"{tag}_feature_names.json", "w") as fh:
                    json.dump(data["feature_names"], fh, indent=2)
            self._log(f"  SHAP saved: {len(results)} diseases → {shap_dir}")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# §3  BATCH DRIVER (scores each row of a personas CSV)
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_strata(
    personas: pd.DataFrame, cli_stratum: str | None, feature_set: str
) -> pd.Series:
    """Resolve one stratum per row: CLI, column, or derived from persona id."""
    if cli_stratum:
        return pd.Series([cli_stratum] * len(personas), index=personas.index)
    for col in _STRATUM_COLS:
        if col in personas.columns:
            return personas[col].astype(str)

    id_col = next((c for c in _ID_CANDIDATES if c in personas.columns), None)
    if id_col is not None:
        derived = personas[id_col].map(
            lambda v: stratum_from_persona_id(v, feature_set)
        )
        if derived.notna().all():
            return derived
        bad = personas.loc[derived.isna(), id_col].tolist()
        raise SystemExit(
            f"Could not derive stratum from persona_id for: {bad}. "
            f"Pass --stratum or add a 'stratum' column."
        )

    raise SystemExit(
        "No stratum given. Pass --stratum, add a 'stratum' column, or provide "
        "persona_ids like '..._F_45' plus --feature_set."
    )


def predict_personas(
    personas_csv: Path,
    cli_stratum: str | None,
    feature_set: str,
    out_path: Path | None,
    project_root: Path | None,
    artifacts_dir: Path | None,
    models_dir: Path | None,
    train_dir: Path | None,
    emit_raw: bool,
    exclude: set[str] | None = None,
    include_excluded: bool = False,
    explain: bool = False,
    explain_dir: Path | None = None,
) -> pd.DataFrame:
    personas = pd.read_csv(personas_csv)
    print(f"[{_ts()}] Loaded {len(personas)} personas from {personas_csv}")

    strata = _resolve_strata(personas, cli_stratum, feature_set)
    frames: list[pd.DataFrame] = []
    cache: dict[str, StagePredictor] = {}

    for stem, idx in personas.groupby(strata).groups.items():
        block = personas.loc[idx]
        print(f"[{_ts()}] === stratum {stem}: {len(block)} persona(s) ===")
        if stem not in cache:
            cache[stem] = StagePredictor(
                stratum=stem,
                project_root=project_root,
                artifacts_dir=artifacts_dir,
                models_dir=models_dir,
                train_dir=train_dir,
                exclude=exclude,
                include_excluded=include_excluded,
            )
        preds = cache[stem].predict(block, emit_raw=emit_raw)
        preds.index = block.index  # keep original CSV order
        frames.append(preds)

        if explain:
            shap_out = explain_dir if explain_dir else (
                (out_path.parent if out_path else personas_csv.parent) / stem
            )
            cache[stem].explain(block, out_dir=shap_out)

    result = pd.concat(frames).sort_index().reset_index(drop=True)

    if out_path is None:
        out_path = personas_csv.with_name("predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[{_ts()}] Wrote {len(result)} rows × {result.shape[1]} cols → {out_path}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §4  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    root = _find_project_root()
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="Artifact-only local inference for STAGE production models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--personas",
        type=Path,
        default=root / "prep_deploy" / "test_personas.csv",
        help="CSV of personas (engineered descriptive-feature space); one row each.",
    )
    p.add_argument(
        "--stratum",
        default=None,
        help="Stratum stem for ALL rows (e.g. Male_all_srdc_poc). Omit to route "
        "per row via a 'stratum' column in the CSV.",
    )
    p.add_argument(
        "--feature_set",
        default="srdc_poc",
        choices=_VALID_FEATURE_SETS,
        help="Feature set used when deriving stratum from persona_id.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: predictions.csv next to --personas).",
    )
    p.add_argument(
        "--project_root",
        type=Path,
        default=None,
        help="Repo root (default: auto-detect via .stage_root).",
    )
    p.add_argument(
        "--artifacts_dir",
        type=Path,
        default=None,
        help="Override preprocessing artifacts root.",
    )
    p.add_argument(
        "--models_dir",
        type=Path,
        default=None,
        help="Override c4_production outputs root.",
    )
    p.add_argument(
        "--train_dir",
        type=Path,
        default=None,
        help="Root containing the TUNED train/ tree (default: "
        "<project_root>/outputs/c4_production_tuned).",
    )
    p.add_argument(
        "--emit_raw",
        action="store_true",
        help="Also emit uncalibrated probabilities where no calibrator exists.",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Extra disease names to withhold from output (added to built-in "
        "deployment exclusions).",
    )
    p.add_argument(
        "--include_excluded",
        action="store_true",
        help="Research mode: ignore built-in deployment exclusions "
        f"({', '.join(sorted(DEPLOY_EXCLUDED_DISEASES))}) and score every "
        "trained disease.",
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="Compute per-sample SHAP values for each disease. Outputs saved "
        "alongside predictions as .npy + .json files in a shap/ subdirectory.",
    )
    p.add_argument(
        "--explain_dir",
        type=Path,
        default=None,
        help="Directory for SHAP outputs (default: shap/ next to --out).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.personas.exists():
        raise SystemExit(f"Personas CSV not found: {args.personas}")
    predict_personas(
        personas_csv=args.personas,
        cli_stratum=args.stratum,
        feature_set=args.feature_set,
        out_path=args.out,
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        models_dir=args.models_dir,
        train_dir=args.train_dir,
        emit_raw=args.emit_raw,
        exclude=set(args.exclude) if args.exclude else None,
        include_excluded=args.include_excluded,
        explain=args.explain,
        explain_dir=args.explain_dir,
    )


if __name__ == "__main__":
    main()