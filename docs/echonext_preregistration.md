# EchoNext structural heart disease: direct FairECG classifier vs NT-proBNP surrogate — pre-specification

Fixed 2026-10-14, after input verification and before any model is scored on EchoNext. Any later change is reported as a declared deviation.

## 0. Question and scope
Does a FairECG classifier trained directly on echocardiographic labels detect structural heart disease (SHD) better than predicted NT-proBNP from the FairECG + HARVI-heads model? And does predicted NT-proBNP add information to the direct classifier?

**Scope.** EchoNext has echo findings but no diagnoses or symptoms. Its positives therefore meet the *structural* criterion of ACC/AHA Stage B, but they mix Stage B with Stage C. Results are reported as **echo-defined structural heart disease**, not as Stage B. The outpatient subgroup is reported as the closest available proxy. A Stage B analysis proper needs symptom status (KARDA).

**Expected asymmetry, stated in advance.** The direct classifier is trained on EchoNext's own labels at the same site. The surrogate is applied zero-shot from MIMIC-IV. The direct model is therefore expected to have the higher AUROC. The informative quantities are (i) the size of the difference, (ii) how many labelled patients a direct model needs to match the surrogate, and (iii) whether the surrogate is complementary.

## 1. Data (verified 2026-10-14)
- EchoNext (Columbia), 100,000 ECGs, 36,286 patients; official splits: train 72,475 ECGs / 26,218 patients, val 4,626 / 4,626, test 5,442 / 5,442.
- Train, val and test are patient-disjoint. `no_split` (17,457 ECGs) shares patients with val and test and is **excluded** throughout.
- Not part of FairECG pretraining. The `fairecgv2clin` corpus is CODE, MGH and MIMIC, and contains 0 EchoNext records.
- Test prevalence: SHD 42.6 %, LVEF ≤ 45 % 17.7 %, LV wall thickness ≥ 13 mm 19.5 %. Test outpatients: 1,059 ECGs, 271 with SHD.
- Waveforms are released per-lead standardised. FairECG applies per-lead instance normalisation (`input_norm='instance'`, verified in the checkpoint), so it is unaffected. The original HARVI-ECG depends on absolute voltage and is **not evaluated**. This corrects the earlier HARVI deviation 2, which wrongly stated that both models depend on absolute voltage.

## 2. Endpoints
- **Primary:** composite SHD moderate-or-greater (`shd`), full test split.
- **Secondary:** LVEF ≤ 45 % (`lvef_le45`); LV wall thickness ≥ 13 mm (`lvwt_ge13`); every endpoint in the outpatient test subgroup.
- **Post hoc (labelled as such):** LVEF ≤ 40 % vs ≥ 50 % (40–50 % excluded), surrogate only. This is the EchoNext check from the HARVI pre-specification §5, run now that the voltage objection is known not to apply to FairECG. It is post hoc because the HARVI results were already known.

## 3. Scores compared
- **S — NT-proBNP surrogate.** The five saved FairECG + HARVI-heads LoRA fold models (MIMIC), applied unchanged. Inputs are age and sex as in training, with the female coding verified against the MIMIC convention. Normalisation constants are recomputed from each fold's training split; this was verified to reproduce saved fold-0 predictions to 0.002 in probability. Score = predicted NT-proBNP, mean of the five folds (as in the Emory analysis). The mean rule-in probability is reported as a secondary surrogate score.
- **D — direct classifier.** FairECG-clinical, the same checkpoint, fine-tuned with LoRA through the benchmark harness `fair_lora.py`, unchanged except for the endpoint definitions.
  - Recipe: rank 8, 20 epochs, ERM, lr 1e-4, batch 32. The full training split is used (train_cap ≥ 72,475) and test_cap covers the full test split.
  - The harness selects the epoch on the validation split's worst age-band AUROC (its fairness-benchmark convention). This works against D on overall AUROC and is declared as such.
  - Seeds 0, 1 and 2. Score = mean probability over the three seeds; the per-seed spread is reported.
  - The harness head has no age/sex input, whereas S has one. An age + sex-only logistic baseline (fit on train, scored on test) is therefore reported alongside.
- **L — label efficiency.**
  - Frozen logistic-regression probes on the cached FairECG-clinical EchoNext embeddings. They use training patients only, with n = 250, 500, 1,000, 2,500, 5,000, 10,000 and all 26,218 patients (all ECGs of each sampled patient), 5 random draws per n (seeds 0–4).
  - The L2 strength C ∈ {1e-3, 1e-2, 1e-1, 1, 10} is chosen on val by AUROC; scoring is on test.
  - LoRA cells (harness, seed 0) at 1,000 and 5,000 patients check that the frozen curves do not understate fine-tuning.
- **C — combination.** Logistic regression on val with two features, D's logit and S (standardised). Fit on val, scored on test.

## 4. Statistics and decision rules
- AUROC with 95 % CIs from 2,000 bootstrap draws (seed 0) over test patients; test has one ECG per patient. Paired differences are computed on identical rows.
- **"D better than S"**: the paired 95 % CI of AUROC(D) − AUROC(S) on primary SHD in the full test split excludes 0.
- **"S adds information to D"**: the paired 95 % CI of AUROC(C) − AUROC(D) on primary SHD excludes 0.
- **Crossover**: the smallest n at which the mean AUROC of the frozen probe over 5 draws is ≥ AUROC(S). This is descriptive; the across-draw range is reported.
- Secondary endpoints and the outpatient subgroup are reported with the same statistics, without multiplicity correction, and interpreted descriptively.

## 5. Not done
~~No threshold-based claims for S on EchoNext~~ (superseded by Amendment 1). No recalibration of S on EchoNext (AUROC is invariant to monotone recalibration). EchoNext `no_split` records are not used.

## Amendment 1 — threshold-based detection (added 2026-10-14, before any EchoNext outcome was scored)
Added at the investigators' request; the surrogate scores existed at this point, but no AUROC or other outcome metric had been computed on EchoNext.

**Endpoints for threshold-based detection (surrogate S only):** LVEF ≤ 40 % vs ≥ 50 % (40–50 % excluded; echo-confirmed HFrEF-range systolic dysfunction), LVEF ≤ 45 %, and composite SHD. Full test split and outpatient subgroup. HFpEF cannot be defined on EchoNext (no symptoms, no diastolic measures) and is not claimed.

**Operating points:**
1. *Fixed MIMIC thresholds, unchanged:* predicted NT-proBNP ≥ 300 pg/mL (rule-out cut) and > age-adjusted rule-in cut (450 / 900 / 1,800 pg/mL for < 50 / 50–75 / > 75 y); also the rule-in head probability ≥ 0.5.
2. *Site-recalibrated thresholds:* on the EchoNext val split, choose the predicted-NT-proBNP threshold that reaches sensitivity ≥ 0.90 (rule-out style) and the one reaching specificity ≥ 0.90 (rule-in style), separately per endpoint, then apply unchanged to test.

**Metrics:** sensitivity, specificity, PPV, NPV, and the fraction flagged positive, with 95 % patient-bootstrap CIs (2,000 draws, seed 0) on test. These are descriptive; no decision rule is attached. Fixed-threshold results are reported as a test of transfer across sites, not as validated cut-offs.
