# Synthetic cohort provenance

Synthetic features were generated on 2026-08-11 by a Gaussian copula fitted on
the c3_6_merged training parquets. They depend only on the training data and the
random seed, not on any model, so they were not regenerated.

SHAP values and predictions were generated on 2026-09-07 against the deployed
survey-exclusion models in outputs/c4_survey_excl_tuned/, replacing the earlier
c4_production_tuned outputs. Predictions come from deploy/predict.py — the same
code path the CDS integration uses — so the numbers here match what SRDC sees at
runtime.

Disease sets match deploy/model_card_pipeline.json: 23 diseases for the Female
strata and Male_60plus, 21 for Male_under60. Substance Use Disorder is scored
through the augmented XGBoost chain, all others through LightGBM singletons.

Risk output is rates_indicative: the calibrated probabilities are displayed and
flagged as indicative, not withheld. They are not yet validated for absolute-risk
interpretation — observed-to-expected runs ~0.5-0.6 for hospital-ascertained
conditions on the test partition, an ascertainment deficit rather than model
miscalibration. Multimorbidity counts and pair scores remain rank_only, because
the error compounds across conditions. See the model card's
deployment_policy.risk_output_rationale and multimorbidity.risk_output_rationale.

This package contains NO real UK Biobank participants.
