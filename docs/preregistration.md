# HARVI-ECG numbers — pre-specification (fixed before any model is scored)

Written 2026-10-13 after the Phase 0 inventory and before any external prediction or FairECG fold model exists.
Any later deviation is listed in the technical report as a declared deviation.

## 1. Internal development data (MIMIC-IV)
- 18,036 ECG–NT-proBNP pairs, 11,929 patients, from the harvi_v2 cache (12 × 5,000 samples, 500 Hz, float32).
- Folds: `GroupKFold(n_splits=5)` on subject_id, exactly as in harvi_v2. Validation sizes reproduce the training logs: [3608, 3607, 3607, 3607, 3607].
- NT-proBNP median 2046 pg/mL (IQR 515–6276); 18.1 % < 300 pg/mL.
- Heads and labels as in `configs/targets/ntprobnp.yaml`: rule-out < 300 pg/mL; age-adjusted rule-in (450 / 900 / 1,800 pg/mL for < 50 / 50–75 / > 75 y); severe > 3,000 pg/mL.
- **Pretraining exposure of FairECG:** 9,506 of 11,929 HARVI patients (14,362 of 18,036 pairs) are in FairECG's self-supervised pretraining corpus (ECGs only, no NT-proBNP or any label). Disclosed; FairECG results are additionally reported on the 2,423 unexposed patients.

## 2. Models compared
- **HARVI-ECG**: the five existing harvi_v2 fold checkpoints, inference only.
- **FairECG + HARVI heads**: FairECG-clinical backbone with LoRA rank 8, the same three classification heads plus the continuous log-NT-proBNP head with uncertainty, the same loss weights and pos_weights, the same folds; age and sex given as in HARVI-ECG. Frozen-probe variant of all three FairECG checkpoints as an ablation.

## 3. Primary internal metric and model-selection rule
- Out-of-fold AUROC for each head, and their mean over rule-out and age-adjusted rule-in (the proposal's "0.846" definition), with patient-bootstrap 95 % CIs (2,000 draws).
- **Reported model** = higher mean OOF AUROC (rule-out + rule-in). If the paired 95 % CI of the difference includes 0, the proposal keeps HARVI-ECG (incumbent, no architecture text change).

## 4. External cohort: Emory (HEEDB site I0006)
Outside FairECG pretraining and HARVI training. ICD-10 phenotypes per patient:
- **HFrEF**: only I50.2x codes; **HFpEF**: only I50.3x codes; **mixed** (I50.4x or both subtypes): included in any-HF only; patients with only I50.1/I50.8/I50.9 excluded.
- **Case ECG**: the patient's ECG closest to the first subtype-specific code, within −180 to +30 days. Available: HFrEF 5,641, HFpEF 5,053, mixed 7,385.
- **Controls**: patients with no I50 code ever and ≥ 365 days follow-up after the ECG; one random ECG per patient (seed 0); available 203,453.
- **Primary**: controls matched 4:1 on sex and 5-year age band, separately for HFrEF and HFpEF (cases are older: median 68 vs 59 y). Secondary: all controls, unmatched.
- **Score**: predicted NT-proBNP, mean of the five fold models, applied unchanged (zero-shot).
- **Endpoints**: AUROC (patient bootstrap 95 % CI) for HFrEF vs control and HFpEF vs control; HFpEF vs HFrEF reported descriptively; sensitivity and specificity at the 300 pg/mL rule-out and the age-adjusted rule-in thresholds.

## 5. Secondary external check: EchoNext (echo-measured LVEF)
LVEF ≤ 40 % vs ≥ 50 % (19,939 vs 64,204 ECGs with LVEF), zero-shot, one ECG per patient. Echo-confirmed complement to the ICD-based HFrEF definition. FairECG never saw EchoNext.

## 6. [Removed for publication]
Internal note on where the numbers are placed in a grant application; contains no analysis content. All other sections are unchanged from the version fixed on 2026-10-13.

## 7. Declared deviations (all fixed before any external prediction existed)
1. **Emory limb leads corrected.** In the HEEDB I0006 WFDB files, the channels labelled aVL and aVF equal (I − II)/2 and (II − I)/2, i.e. −III/2 and +III/2 (least-squares fit on 300 records, residual 0.6 %; leads I, II, III, aVR verified by Einthoven). Before scoring, aVL and aVF are rebuilt as I − II/2 and II − I/2 for every Emory ECG; after the fix the Einthoven residual is < 1e-7.
2. **EchoNext dropped.** The released EchoNext waveforms are per-lead standardised (every lead SD ≈ 0.87), not in mV. Both models depend on absolute voltage, so a zero-shot test there is not valid. The secondary echo-LVEF check is therefore not reported. *Later correction (14 October 2026):* this applies to the task-specific baseline only. The FairECG-based readout normalises each lead itself, so the voltage objection does not apply to it; the echo-LVEF check was therefore run for that readout under a separate pre-specification ([docs/echonext_preregistration.md](echonext_preregistration.md)), where it is labelled post hoc.
3. **Matched sample sizes.** After excluding ages outside 18–100 or missing: HFrEF 5,627 cases / 22,508 controls; HFpEF 5,045 / 20,180; no stratum shortfall.
4. **Epoch selection differs between the two models.** HARVI-ECG's existing checkpoints were selected (early stopping, best checkpoint) on `val_avg_auc` computed on the evaluation fold itself, as in the harvi_v2 training script. FairECG selects its epoch on an inner 10 % of training patients. This biases the internal comparison in HARVI-ECG's favour and is reported as such; the external test is unaffected because neither model was selected on Emory.
5. **FairECG training recipe.** No SupCon loss, mixup, lead dropout or random crops (HARVI-ECG uses them); the paper campaign's LoRA recipe instead. Age and sex enter a 2-layer head rather than FiLM conditioning.
