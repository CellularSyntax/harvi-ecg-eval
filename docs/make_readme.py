"""Generate README.md from the result tables (no hand-typed numbers).  Usage: python docs/make_readme.py"""
import os, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = pd.read_csv(f"{ROOT}/results/harvi_internal_results.csv"); E = pd.read_csv(f"{ROOT}/results/harvi_external_results.csv")
X = pd.read_csv(f"{ROOT}/results/echonext/echonext_surrogate_auroc.csv")
XD = pd.read_csv(f"{ROOT}/results/echonext/echonext_direct_vs_surrogate.csv"); XL = pd.read_csv(f"{ROOT}/results/echonext/echonext_label_efficiency_summary.csv")
A, U = "MIMIC OOF, all", "MIMIC OOF, unexposed to FairECG pretraining"
BASE, HARVI = "HARVI-ECG", "FairECG-clin LoRA"          # internal labels: task-specific baseline, HARVI-ECG
MO = "mean rule-out+rule-in"


def one(df, **kw):
    s = df
    for k, v in kw.items(): s = s[s[k] == v]
    assert len(s) == 1, (kw, len(s)); return s.iloc[0]


g = lambda st, m, h: one(R, set=st, model=m, head=h)
ex = lambda m, e: one(E, model=m, endpoint=e)
xs = lambda sub, ep: one(X, subset=sub, endpoint=ep, score="predicted NT-proBNP")
xd = lambda sub, ep: one(XD, subset=sub, endpoint=ep)
ci = lambda v, lo, hi: f"**{v:.3f}** <sub>({lo:.3f}–{hi:.3f})</sub>"
cr = lambda r: ci(r.auroc, r.lo, r.hi)
d = lambda r: f"{r.d_vs_harvi:+.3f} <sub>({r.d_lo:+.3f} to {r.d_hi:+.3f})</sub>"
n = lambda v: f"{int(v):,}"

mo, hmo, umo = g(A, HARVI, MO), g(A, BASE, MO), g(U, HARVI, MO)
ro, ri, sv = (g(A, HARVI, h) for h in ("rule-out", "rule-in (age-adj.)", "severe"))
efr, efp = ex(HARVI, "HFREF vs matched controls"), ex(HARVI, "HFPEF vs matched controls")
befr, befp = ex(BASE, "HFREF vs matched controls"), ex(BASE, "HFPEF vs matched controls")
EP = [("SHD (composite)", "Structural heart disease (composite)"), ("LVEF <=45%", "LVEF ≤ 45 %"),
      ("LVEF <=40% vs >=50% (post hoc)", "Reduced EF (LVEF ≤ 40 % vs ≥ 50 %) †"), ("LV wall >=13 mm", "LV wall thickness ≥ 13 mm")]
x40 = xd("all test", EP[2][0]); x45 = xd("all test", EP[1][0]); xshd = xd("all test", EP[0][0])
le = lambda ep, k: one(XL, endpoint=ep, n_patients=k)
cross = {}
for ep in ("lvef_le45", "shd"):
    s = XL[XL.endpoint == ep].sort_values("n_patients"); cross[ep] = int(s[s["mean"] >= s.surrogate_auroc].n_patients.min())
n_test = int(xshd.n)

L = []
L += ['<p align="center">', '  <img src="docs/assets/harvi_ecg_logo.png" alt="HARVI-ECG">', '</p>', '',
      '<p align="center">', '  <b>Heart-failure readouts from a standard 10-second 12-lead ECG</b>', '</p>', '',
      '<p align="center">', '  <a href="https://cellularsyntax.github.io/harvi-ecg-eval/"><b>Explore the interactive project website →</b></a>', '</p>', '',
      '<p align="center">',
      '  <a href="https://cellularsyntax.github.io/harvi-ecg-eval/"><img src="https://img.shields.io/badge/project-website-6A00A8" alt="Project website"></a>',
      '  <a href="report/harvi_results_report.pdf"><img src="https://img.shields.io/badge/technical%20report-PDF-0F6E78" alt="Technical report"></a>',
      '  <a href="https://doi.org/10.5281/zenodo.23367346"><img src="https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23367346-1682D4" alt="Zenodo DOI"></a>',
      '  <a href="https://huggingface.co/maxhaberbusch/HARVI-ECG"><img src="https://img.shields.io/badge/Hugging%20Face-weights%20on%20request-FFD21E?logo=huggingface&logoColor=black" alt="Model weights on Hugging Face (on request)"></a>',
      '  <a href="docs/preregistration.md"><img src="https://img.shields.io/badge/analysis-pre--specified-546E7A" alt="Pre-specified analysis"></a>',
      '  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-8D6E63" alt="MIT licence"></a>', '</p>', '',
      '---', '']
L += ["## Why", "",
      "Heart failure is often recognised late. NT-proBNP is the guideline blood test for ruling heart failure in or out, and echocardiography confirms structural disease, but neither is done routinely in people without symptoms. A 12-lead ECG is cheap, quick and recorded millions of times a day. **HARVI-ECG** reads heart-failure information out of that ECG, so the patients who need a blood test or an echo can be identified from a recording that already exists.", "",
      '<p align="center"><img src="docs/assets/harvi_visual_abstract.png" alt="HARVI-ECG graphical abstract"></p>', "", ""]
L += ["## At a glance", "",
      "| | |", "|---|---|",
      f"| **Rule out** NT-proBNP < 300 pg/mL (MIMIC-IV, 5-fold out-of-fold) | AUROC {cr(ro)} |",
      f"| **Detect HFrEF** at a hospital it has never seen (Emory, zero-shot) | AUROC {cr(efr)} |",
      f"| **Detect reduced ejection fraction** at a second unseen hospital (EchoNext, zero-shot) | AUROC {cr(xs('all test', EP[2][0]))} |",
      f"| **Detect reduced ejection fraction** after adaptation on local echo labels (EchoNext held-out test) | AUROC {ci(x40.direct_auroc, x40.direct_lo, x40.direct_hi)} |",
      f"| Echo-labelled patients needed to match the zero-label readout (LVEF ≤ 45 %) | **{cross['lvef_le45']}** |", ""]
L += ["## How it works", "",
      '<p align="center"><img src="docs/assets/harvi_fig1_overview.png" alt="HARVI-ECG architecture and cohorts"></p>', "",
      "- **ECG foundation encoder (FairECG).** An xLSTM-based encoder pretrained without labels (SimDINO) on 6,009,279 ECGs from 3,257,195 patients (CODE, MGH, MIMIC-IV-ECG). It turns a 10-second, 500 Hz, 12-lead ECG into a 1,024-dimensional embedding.",
      "- **NT-proBNP readout.** The encoder is adapted with low-rank adapters (LoRA, rank 8) and feeds a head that also receives age and sex. It outputs an NT-proBNP estimate and three guideline decisions: rule-out (< 300 pg/mL), age-adjusted rule-in (> 450 / 900 / 1,800 pg/mL for < 50 / 50–75 / > 75 years) and severe (> 3,000 pg/mL). Trained on MIMIC-IV only.",
      "- **Structural heart disease (SHD) module.** The same encoder with one output per echocardiographic endpoint, adapted on echo-labelled patients from the target site.", ""]
L += ["## The NT-proBNP readout works and transfers", "",
      '<p align="center"><img src="docs/assets/harvi_fig2_ntprobnp.png" alt="NT-proBNP readout results"></p>', "",
      "| Cohort | Endpoint | ECGs | AUROC (95 % CI) |", "|---|---|---|---|",
      f"| MIMIC-IV, 5-fold OOF | Rule-out < 300 pg/mL | {n(ro.n)} | {cr(ro)} |",
      f"| | Rule-in (age-adjusted) | {n(ri.n)} | {cr(ri)} |",
      f"| | Severe > 3,000 pg/mL | {n(sv.n)} | {cr(sv)} |",
      f"| | **Mean rule-out + rule-in** (pre-specified) | | {cr(mo)} |",
      f"| Emory, zero-shot | HFrEF vs matched controls | {n(efr.n_cases + efr.n_controls)} | {cr(efr)} |",
      f"| | HFpEF vs matched controls | {n(efp.n_cases + efp.n_controls)} | {cr(efp)} |",
      f"| EchoNext, zero-shot | Reduced EF † | {n(x40.n)} | {ci(xs('all test', EP[2][0]).auroc, xs('all test', EP[2][0]).lo, xs('all test', EP[2][0]).hi)} |",
      f"| | LVEF ≤ 45 % | {n(x45.n)} | {ci(xs('all test', EP[1][0]).auroc, xs('all test', EP[1][0]).lo, xs('all test', EP[1][0]).hi)} |", "",
      f"MIMIC-IV: {n(ro.n)} ECGs from {n(ro.n_pat)} patients. Patients whose ECG waveforms never entered pretraining ({n(umo.n_pat)}) score similarly: mean rule-out + rule-in {cr(umo)}. Emory labels are ICD-10 coded; controls are matched 4:1 on age and sex.", ""]
L += ["## Foundation pretraining beats a task-specific model", "",
      '<p align="center"><img src="docs/assets/harvi_fig3_foundation.png" alt="HARVI-ECG vs task-specific baseline"></p>', "",
      f"On identical patients and folds, HARVI-ECG improves on a CNN-xLSTM trained from scratch for the task: {mo.auroc:.3f} vs {hmo.auroc:.3f} on the pre-specified MIMIC-IV metric (Δ {d(mo)}), and most at an unseen hospital: Emory HFrEF {efr.auroc:.3f} vs {befr.auroc:.3f} (Δ {d(efr)}), HFpEF {efp.auroc:.3f} vs {befp.auroc:.3f} (Δ {d(efp)}).", ""]
L += ["## The SHD module detects structural heart disease with few local labels", "",
      '<p align="center"><img src="docs/assets/harvi_fig4_shd.png" alt="SHD module results"></p>', "",
      f"EchoNext held-out test patients (n = {n_test:,}, one ECG each). The SHD module is adapted on EchoNext's echo-labelled training patients; the NT-proBNP readout is applied without any EchoNext labels.", "",
      "| Endpoint | Cases | NT-proBNP readout (zero-shot) | SHD module (adapted) | Δ |", "|---|---|---|---|---|"]
for ep, nm in EP:
    r = xd("all test", ep); s = xs("all test", ep)
    L.append(f"| {nm} | {n(r.n_pos)} | {ci(s.auroc, s.lo, s.hi)} | {ci(r.direct_auroc, r.direct_lo, r.direct_hi)} | {r.d_direct_minus_surrogate:+.3f} |")
L += ["", f"With a lightweight adaptation of the encoder, **{cross['lvef_le45']} echo-labelled patients** match the zero-label NT-proBNP readout for LVEF ≤ 45 % ({n(cross['shd'])} for structural heart disease), and accuracy keeps rising with more labels ({le('lvef_le45', 26218)['mean']:.3f} with all {n(26218)} training patients).", "",
      "† Post hoc endpoint; LVEF ≤ 45 % is the pre-specified alternative.", ""]
L += ["## Comparison with published models", "",
      '<p align="center"><img src="docs/assets/harvi_benchmark.png" alt="HARVI-ECG against published ECG-to-NT-proBNP models"></p>', "",
      "Values for published models as reported in the cited studies; they were evaluated on different cohorts.", ""]
L += ["## Study design", "",
      "- **Pre-specified.** The decision metric, statistics and external endpoints were fixed before any scoring ([HARVI pre-specification](docs/preregistration.md), [EchoNext pre-specification](docs/echonext_preregistration.md)). 95 % CIs from a 2,000-draw patient bootstrap; paired differences on identical rows.",
      "- **Patient-level separation.** MIMIC-IV folds are grouped by patient; Emory and EchoNext were used neither for development nor for pretraining; EchoNext training, validation and test splits are patient-disjoint.",
      "- **Zero-shot means zero-shot.** At Emory and EchoNext the NT-proBNP readout is applied exactly as trained on MIMIC-IV (mean of the five fold models), with no recalibration.", ""]
L += ["## Limitations", "",
      "- Retrospective data; prospective validation of the screening pathway is the next step.",
      "- Emory phenotypes are ICD-10 based. EchoNext records echo findings but not symptoms, so its positives include symptomatic heart failure as well as asymptomatic structural disease.",
      "- AUROC transfers between hospitals; absolute pg/mL cut-offs need recalibration at a new site.", ""]
L += ["## Repository layout", "", "```",
      "src/        data inventory, external cohorts, model training and inference (numbered in run order)",
      "src/checks/ lead-order / lead-integrity checks and evaluation diagnostics",
      "slurm/      cluster batch scripts (paths parameterised)",
      "scoring/    pre-specified scoring (MIMIC-IV, Emory, EchoNext)",
      "results/    aggregate results only (no patient-level data)",
      "report/     LaTeX technical report: generator, source, figures, PDF",
      "docs/       pre-specifications, README generator, assets", "```", ""]
L += ["## Reproducing", "",
      "See [REPRODUCE.md](REPRODUCE.md). Scoring is deterministic (seed 0): re-running it on the exported predictions reproduces `results/*.csv` byte-for-byte. This README is generated from those tables by `docs/make_readme.py`.", ""]
L += ["## Data access", "",
      "No patient-level data, predictions or model weights are distributed. **MIMIC-IV / MIMIC-IV-ECG** and the **Harvard-Emory ECG Database** are credentialed-access datasets on PhysioNet. **EchoNext** is available under its own data-use agreement.", ""]
L += ["## Links", "",
      "- Project website: <https://cellularsyntax.github.io/harvi-ecg-eval/>",
      "- Technical report: [report/harvi_results_report.pdf](report/harvi_results_report.pdf)",
      "- Model weights: [Hugging Face, maxhaberbusch/HARVI-ECG](https://huggingface.co/maxhaberbusch/HARVI-ECG), available upon reasonable request (access is reviewed and approved manually)",
      "- Archived release (report, results, code): Zenodo, DOI [10.5281/zenodo.23367346](https://doi.org/10.5281/zenodo.23367346) (resolves once the record is published)",
      "- Pre-specifications: [HARVI](docs/preregistration.md) · [EchoNext](docs/echonext_preregistration.md)", ""]
L += ["## Citation and licence", "",
      "See [CITATION.cff](CITATION.cff). Code: MIT ([LICENSE](LICENSE)). Report, figures and aggregate results: CC BY 4.0.", ""]
open(f"{ROOT}/README.md", "w").write("\n".join(L))
print("README.md written")
