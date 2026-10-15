"""Build report/harvi_results_report.tex from the result CSVs and prediction files (no hand-typed numbers).
Usage: HARVI_DATA=/path/to/exports python report/build_report.py && cd report && pdflatex harvi_results_report.tex (twice)"""
import numpy as np, pandas as pd, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
DATA = os.environ.get("HARVI_DATA", "data")
RES_INT, RES_EXT = "results/harvi_internal_results.csv", "results/harvi_external_results.csv"
from scipy.stats import spearmanr
R = pd.read_csv(RES_INT); E = pd.read_csv(RES_EXT)
D = pd.read_csv(f"{DATA}/harvi_oof_merged.csv"); Z = np.load(f"{DATA}/score_export.npz"); EM = pd.read_csv(f"{DATA}/score_emory_meta.csv")
A, U = "MIMIC OOF, all", "MIMIC OOF, unexposed to FairECG pretraining"
H, F = "HARVI-ECG", "FairECG-clin LoRA"
g = lambda s, m, h: R[(R.set == s) & (R.model == m) & (R["head"] == h)].iloc[0]
ex = lambda m, e: E[(E.model == m) & (E.endpoint == e)].iloc[0]
f3 = lambda x: f"{x:.3f}"; s3 = lambda x: f"{x:+.3f}".replace("-", "$-$")
ci = lambda r: f"{r.auroc:.3f} ({r.lo:.3f}--{r.hi:.3f})"
cc = lambda r: r"\makecell{" + f"{r.auroc:.3f}" + r"\\ {\scriptsize " + f"{r.lo:.3f}--{r.hi:.3f}" + r"}}"
dc = lambda r: r"\makecell{" + s3(r.d_vs_harvi) + r"\\ {\scriptsize " + f"{s3(r.d_lo)} to {s3(r.d_hi)}" + r"}}"
dci = lambda r: f"{s3(r.d_vs_harvi)} ({s3(r.d_lo)} to {s3(r.d_hi)})"
n = lambda x: f"{int(x):,}".replace(",", "{,}")
MO = "mean rule-out+rule-in"; HE = ["rule-out", "rule-in (age-adj.)", "severe", MO]
ro, ri, sv, mo = (g(A, F, h) for h in HE); hmo, hro = g(A, H, MO), g(A, H, "rule-out")
umo = g(U, F, MO)
efr, efp = ex(F, "HFREF vs matched controls"), ex(F, "HFPEF vs matched controls")
ehr, ehp = ex(H, "HFREF vs matched controls"), ex(H, "HFPEF vs matched controls")
# cohort + calibration (recomputed from predictions)
y = D.ntprobnp.values; age = D.age.values; thr = np.where(age < 50, 450, np.where(age > 75, 1800, 900))
q = lambda v, p: f"{np.percentile(v, p):,.0f}".replace(",", "{,}")
mu, hp = Z["fairecgv2clin_lora_mu"], D.harvi_pred_ntprobnp.values
def slope(p):
    lp = np.log10(np.clip(p, 1, None)); return np.polyfit(lp, np.log10(np.clip(y, 1, None)), 1)[0]
cal = dict(rF=spearmanr(mu, y)[0], rH=spearmanr(hp, y)[0], sF=slope(mu), sH=slope(hp), negF=100 * (mu <= 0).mean(), negH=100 * (hp <= 0).mean(),
           biasH=(np.log10(np.clip(hp, 1, None)) - np.log10(np.clip(y, 1, None))).mean())
def emc(ph):
    c = EM[EM.label == ph]; k = EM[(EM.label == "control") & (EM.ctl_for == ph)]
    return c.age.median(), k.age.median(), 100 * c.female.mean(), 100 * k.female.mean()
cr, cp = emc("hfref"), emc("hfpef")
n_pre = int((D.drop_duplicates("subject_id").in_fairecg_pretraining == 1).sum()); n_pat = D.subject_id.nunique()

def int_table(s, cap, lab):
    rows = []
    for m, nm in ((H, "Task-specific baseline"), (F, r"\textbf{HARVI-ECG}")):
        c = [g(s, m, h) for h in HE]
        rows.append(f"{nm} & " + " & ".join(cc(x) for x in c) + " & " + ("---" if m == H else dc(c[3])) + r" \\ \addlinespace[2pt]")
    r0 = g(s, H, MO)
    return (r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}" "\n"
            rf"\caption{{{cap} ($n$ = {n(r0.n)} ECGs, {n(r0.n_pat)} patients). AUROC with 95\,\% patient-bootstrap CI (2{{,}}000 draws). $\Delta$ is the paired difference to the task-specific baseline on the pre-specified decision metric.}}\label{{{lab}}}" "\n"
            r"\begin{tabular}{@{}lccccc@{}}\toprule" "\n"
            r"Model & \makecell{Rule-out\\$<$300} & \makecell{Rule-in\\(age-adj.)} & \makecell{Severe\\$>$3{,}000} & \makecell{Mean rule-out\\+ rule-in} & \makecell{$\Delta$ mean vs\\baseline} \\\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}\end{table}" "\n")

ext_rows = []
for e, en in (("HFREF vs matched controls", "HFrEF vs controls"), ("HFPEF vs matched controls", "HFpEF vs controls")):
    for m, nm in ((H, "Baseline"), (F, r"\textbf{HARVI-ECG}")):
        r = ex(m, e)
        ext_rows.append(f"{en if m == H else ''} & {nm} & {cc(r)} & {'---' if m == H else dc(r)} & {r.sens_at_300:.2f} / {r.spec_at_300:.2f} & {r.sens_at_ruleIn:.2f} / {r.spec_at_ruleIn:.2f}" + r" \\")
    if e.startswith("HFREF"): ext_rows.append(r"\addlinespace")
d_rp_H, d_rp_F = ex(H, "HFrEF vs HFpEF (descriptive)").auroc, ex(F, "HFrEF vs HFpEF (descriptive)").auroc

# ---- EchoNext section (from results/echonext)
XD = pd.read_csv("results/echonext/echonext_direct_vs_surrogate.csv"); XL = pd.read_csv("results/echonext/echonext_label_efficiency_summary.csv")
xd = lambda sub, ep: XD[(XD.subset == sub) & (XD.endpoint == ep)].iloc[0]
xs, x45, x40, xw = (xd("all test", e) for e in ("SHD (composite)", "LVEF <=45%", "LVEF <=40% vs >=50% (post hoc)", "LV wall >=13 mm"))
o45 = xd("outpatient", "LVEF <=45%")
cross = {c: int(XL[(XL.endpoint == c) & XL.probe_ge_surrogate].n_patients.min()) for c in ("shd", "lvef_le45", "lvwt_ge13")}
le = lambda c, n: XL[(XL.endpoint == c) & (XL.n_patients == n)].iloc[0]
dd = lambda r: f"{s3(r.d_direct_minus_surrogate)} ({s3(r.d_lo)} to {s3(r.d_hi)})"
ac = lambda a, lo, hi: r"\makecell{" + f"{a:.3f}" + r"\\ {\scriptsize " + f"{lo:.3f}--{hi:.3f}" + r"}}"
XS = pd.read_csv("results/echonext/echonext_surrogate_auroc.csv")
def sci(r):
    q = XS[(XS.subset == r.subset) & (XS.endpoint == r.endpoint) & (XS.score == "predicted NT-proBNP")].iloc[0]
    assert abs(q.auroc - r.surrogate_auroc) < 1e-9; return ac(q.auroc, q.lo, q.hi)
xrow = lambda nm, r: f"{nm} & {n(r.n_pos)} / {n(r.n)} & {sci(r)} & " + r"\makecell{\textbf{" + f"{r.direct_auroc:.3f}" + r"}\\ {\scriptsize " + f"{r.direct_lo:.3f}--{r.direct_hi:.3f}" + r"}} & \makecell{" + s3(r.d_direct_minus_surrogate) + r"\\ {\scriptsize " + f"{s3(r.d_lo)} to {s3(r.d_hi)}" + r"}} \\ \addlinespace[2pt]"
XROWS = "\n".join([xrow("Structural heart disease$^a$", xs), xrow("LVEF $\\le$45\\,\\%", x45), xrow("LVEF $\\le$40 vs $\\ge$50\\,\\%$^b$", x40), xrow("LV wall $\\ge$13\\,mm", xw)])

r0_ = g(A, F, MO)

# ---- operating points (results/operating_points.csv; thresholds chosen on data separate from the evaluated rows)
OP = pd.read_csv("results/operating_points.csv")
op = lambda pan, pt, ser="HARVI-ECG": OP[(OP.panel == pan) & (OP.point == pt) & (OP.series.str.startswith(ser))].iloc[0]
SER_NT, SER_SHD = "HARVI-ECG · NT-proBNP", "HARVI-ECG · SHD"
YD, DF = "Youden (cross-fitted)", "default"
T300, TRI, RYD, VY = "transferred cut-off 300 pg/mL", "transferred age-adjusted rule-in", "site-recalibrated Youden (cross-fitted)", "Youden (validation split)"
p2 = lambda v: f"{v:.2f}"
ss = lambda r: f"{p2(r.sensitivity)}/{p2(r.specificity)}"
sci2 = lambda v, lo, hi: f"{v:.2f} ({lo:.2f}--{hi:.2f})"
# rule-out decision at the default operating point, recomputed from the out-of-fold predictions
p_ro = Z["fairecgv2clin_lora_p"][:, 0]; pos = y < 300
ro_sens, ro_spec = (p_ro[pos] >= 0.5).mean(), (p_ro[~pos] < 0.5).mean()
assert abs(ro_sens - op("rule_out_lt300", DF).sensitivity) < 1e-3 and abs(ro_spec - op("rule_out_lt300", DF).specificity) < 1e-3

def thr_txt(r):
    if pd.isna(r.threshold): return r"450/900/1{,}800 pg/mL"
    if "pg/mL" in str(r.threshold_unit): return f"{r.threshold:,.0f} pg/mL".replace(",", "{,}")
    return f"$p \\geq {r.threshold:.2f}$"
OPROWS = []
def oprow(label, r): OPROWS.append(f"\\quad {label} & {thr_txt(r)} & {sci2(r.sensitivity, r.sens_lo, r.sens_hi)} & {sci2(r.specificity, r.spec_lo, r.spec_hi)} \\\\")
def ophead(t): OPROWS.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{t}}}}} \\\\")
ophead(r"MIMIC-IV, out-of-fold (NT-proBNP readout)")
for pan, nm in (("rule_out_lt300", "Rule-out"), ("rule_in_age_adjusted", "Rule-in"), ("severe_gt3000", "Severe")):
    oprow(f"{nm}, Youden (cross-fitted)", op(pan, YD)); oprow(f"{nm}, default", op(pan, DF))
OPROWS.append(r"\addlinespace")
ophead(r"Emory, zero-shot (NT-proBNP readout)")
for pan, nm in (("hfref_vs_matched_controls", "HFrEF"), ("hfpef_vs_matched_controls", "HFpEF")):
    oprow(f"{nm}, transferred 300 pg/mL", op(pan, T300)); oprow(f"{nm}, transferred age-adjusted rule-in", op(pan, TRI)); oprow(f"{nm}, recalibrated Youden", op(pan, RYD))
OPROWS.append(r"\addlinespace")
ophead(r"EchoNext, held-out test (Youden threshold from the validation split)")
for pan, nm in (("lvef_le40_vs_ge50", r"Reduced EF$^\dagger$"), ("lvef_le45", r"LVEF $\le$45\,\%"), ("shd", r"SHD composite$^a$"), ("lvwt_ge13", r"LV wall $\ge$13\,mm")):
    oprow(f"{nm}, NT-proBNP readout", op(pan, VY, SER_NT)); oprow(f"{nm}, SHD module", op(pan, VY, SER_SHD))
OPROWS = "\n".join(OPROWS)
oe, oep = op("hfref_vs_matched_controls", TRI), op("hfref_vs_matched_controls", RYD)
o3, o3p = op("hfref_vs_matched_controls", T300), op("hfpef_vs_matched_controls", T300)
xn, xsh = op("lvef_le40_vs_ge50", VY, SER_NT), op("lvef_le40_vs_ge50", VY, SER_SHD)

T = rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,textwidth=5.5in,textheight=9in,top=1in,headsep=0.25in,footskip=0.4in]{{geometry}}
\usepackage[T1]{{fontenc}}\usepackage[utf8]{{inputenc}}\usepackage{{lmodern}}\usepackage{{microtype}}
\usepackage{{amsmath,amssymb}}\usepackage{{makecell,booktabs,graphicx,xcolor,enumitem,float}}
\usepackage[numbers,square,sort&compress]{{natbib}}
\usepackage[font=small,labelfont=bf,labelsep=colon,skip=6pt]{{caption}}
\usepackage{{titlesec}}
\titleformat{{\section}}{{\large\bfseries}}{{\thesection}}{{0.6em}}{{}}
\titleformat{{\subsection}}{{\normalsize\bfseries}}{{\thesubsection}}{{0.6em}}{{}}
\titlespacing*{{\section}}{{0pt}}{{12pt plus 2pt minus 2pt}}{{6pt}}
\titlespacing*{{\subsection}}{{0pt}}{{9pt plus 2pt minus 2pt}}{{4pt}}
\setlength{{\emergencystretch}}{{1.5em}}\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5.5pt}}
\definecolor{{mydarkblue}}{{rgb}}{{0,0.08,0.45}}
\usepackage[colorlinks=true,linkcolor=mydarkblue,citecolor=mydarkblue,urlcolor=mydarkblue]{{hyperref}}
\setlist{{itemsep=1pt,topsep=2pt,leftmargin=1.5em}}
\renewcommand{{\paragraph}}[1]{{\textbf{{#1}}\hspace{{0.5em}}}}
\begin{{document}}
\thispagestyle{{plain}}
\begin{{center}}
\rule{{\linewidth}}{{4pt}}\par\vspace{{0.18in}}
{{\LARGE\bfseries HARVI-ECG: Heart-Failure Readouts from a\\[2pt] 10-Second 12-Lead ECG\par}}
\vspace{{0.12in}}\rule{{\linewidth}}{{1pt}}\par\vspace{{0.22in}}
{{\bfseries Medical University of Vienna}}\\
Center for Medical Physics and Biomedical Engineering\par
\vspace{{0.2in}}
\end{{center}}
\begin{{center}}{{\large\bfseries Abstract}}\end{{center}}\vspace{{-4pt}}
\begin{{quote}}\small
HARVI-ECG turns a standard 10-second 12-lead ECG into heart-failure readouts. It is built on an ECG foundation encoder pretrained without labels on 6{{,}}009{{,}}279 ECGs from 3{{,}}257{{,}}195 patients, and provides two readouts: an \emph{{NT-proBNP readout}} (an NT-proBNP estimate with rule-out, age-adjusted rule-in and severe decisions) and a \emph{{structural heart disease (SHD) module}} that detects echo-defined structural heart disease. On MIMIC-IV ({n(r0_.n)} ECG--NT-proBNP pairs, {n(n_pat)} patients, patient-grouped 5-fold cross-validation), the NT-proBNP readout reached a mean rule-out/rule-in AUROC of {ci(mo)} and a rule-out AUROC of {ci(ro)}. Without retraining it transferred to two hospitals it had never seen: at Emory it detected HFrEF with AUROC {ci(efr)} and HFpEF with {ci(efp)} against age- and sex-matched controls, and at EchoNext it identified echocardiographic reduced ejection fraction with AUROC {x40.surrogate_auroc:.3f}. HARVI-ECG outperformed a task-specific baseline trained from scratch, most clearly at the unseen hospital (Emory HFrEF {dci(efr)}, HFpEF {dci(efp)}). Adapted on EchoNext echocardiography labels, the SHD module reached AUROC {x40.direct_auroc:.3f} for reduced ejection fraction and {xs.direct_auroc:.3f} for composite structural heart disease on held-out patients, and needed only {cross['lvef_le45']} labelled patients to match the zero-label NT-proBNP readout for LVEF $\le$45\,\%.
\end{{quote}}
\vspace{{2pt}}
\begin{{figure}}[h]\centering\includegraphics[width=\linewidth]{{harvi_visual_abstract.png}}
\caption*{{\small\textbf{{Overview.}} A 10-second 12-lead ECG is encoded by a self-supervised ECG foundation model; HARVI-ECG reads out NT-proBNP-based heart-failure risk and echo-defined structural heart disease.}}\end{{figure}}
\renewcommand{{\thefootnote}}{{}}\footnotetext{{Technical report v1.3, October 2026. Code: \url{{https://github.com/CellularSyntax/harvi-ecg-eval}}. Archive: \href{{https://doi.org/10.5281/zenodo.23367346}}{{doi:10.5281/zenodo.23367346}}. Model weights (on request): \url{{https://huggingface.co/maxhaberbusch/HARVI-ECG}}.}}\renewcommand{{\thefootnote}}{{\arabic{{footnote}}}}

\section{{Model}}\label{{sec:model}}
\begin{{figure}}[t]\centering\includegraphics[width=\linewidth]{{harvi_fig1_overview.png}}
\caption{{HARVI-ECG and its cohorts. (a) Architecture; age and sex feed only the NT-proBNP readout. (b) Cohort sites; circle area is proportional to cohort size. The Boston circle marks MIMIC-IV; the pretraining sources MGH and MIMIC-IV-ECG are also Boston-based and are listed in (c). (c) Cohort roles and counts. Emory and EchoNext were used neither in pretraining nor in developing the NT-proBNP readout.}}\label{{fig:overview}}\end{{figure}}
HARVI-ECG (Figure~\ref{{fig:overview}}a) has three parts.
\begin{{itemize}}
\item \textbf{{ECG foundation encoder.}} The FairECG encoder (internal, ongoing unpublished work), an xLSTM-based model~\cite{{beck2024}} pretrained with self-supervision (SimDINO) on 6{{,}}009{{,}}279 ECGs from 3{{,}}257{{,}}195 patients (CODE, MGH, MIMIC-IV-ECG; waveforms only, no labels). It maps a 10-second, 500-Hz, 12-lead ECG (each lead instance-normalised) to a 1{{,}}024-dimensional embedding.
\item \textbf{{NT-proBNP readout.}} The encoder is adapted with low-rank adapters (LoRA, rank 8)~\cite{{hu2022}} and feeds a shared head that also receives age and sex. The head outputs an NT-proBNP estimate with uncertainty and three guideline decisions: rule-out ($<$300\,pg/mL), age-adjusted rule-in ($>$450/900/1{{,}}800\,pg/mL for $<$50/50--75/$>$75 years) and severe ($>$3{{,}}000\,pg/mL). It is trained on MIMIC-IV only; the epoch is selected on an inner 10\,\% of training patients.
\item \textbf{{SHD module.}} The same encoder with LoRA and one logistic output per echocardiographic endpoint, adapted on the echo-labelled patients of a site (here EchoNext).
\end{{itemize}}
A task-specific baseline (ConvNeXt-1D + xLSTM with age and sex conditioning, trained from scratch on the same MIMIC-IV folds) serves as a reference (Section~\ref{{sec:found}}); because it also differs in architecture, training recipe and checkpoint selection, the comparison does not isolate the effect of pretraining alone. Analyses followed pre-specifications fixed before scoring; the primary decision metric was the mean of the rule-out and rule-in AUROCs, with 95\,\% CIs from a 2{{,}}000-draw patient bootstrap.

\section{{Data}}\label{{sec:data}}
\paragraph{{MIMIC-IV (development).}} {n(r0_.n)} ECG--NT-proBNP pairs from {n(n_pat)} patients of the MIMIC-IV-ECG matched subset~\cite{{gow2023,johnson2023}}. Median age {q(age,50)} years (IQR {q(age,25)}--{q(age,75)}); {100*D.female.mean():.1f}\,\% female; NT-proBNP median {q(y,50)}\,pg/mL (IQR {q(y,25)}--{q(y,75)}). Prevalence: rule-out {100*(y<300).mean():.1f}\,\%, rule-in {100*(y>thr).mean():.1f}\,\%, severe {100*(y>3000).mean():.1f}\,\%. Folds are \texttt{{GroupKFold(5)}} on patient.

\paragraph{{Emory (external).}} Site I0006 of the Harvard-Emory ECG Database~\cite{{koscova2026}}, used neither for development nor for pretraining. HFrEF (ICD-10 I50.2x) and HFpEF (I50.3x) cases, with the ECG closest to the first subtype code ($-$180 to $+$30 days), against controls without any I50 code and at least 365 days of follow-up, matched 4:1 on sex and 5-year age band. HFrEF: {n(efr.n_cases)} cases, {n(efr.n_controls)} controls. HFpEF: {n(efp.n_cases)} cases, {n(efp.n_controls)} controls. One ECG per patient (53{{,}}360 in total). The stored aVL and aVF channels were rebuilt from leads I and II before scoring.

\paragraph{{EchoNext (external).}} 12-lead ECGs paired with transthoracic echocardiography (Columbia University), with patient-disjoint training, validation and test splits; not used in pretraining. All EchoNext results refer to the {n(xs.n)} test patients (one ECG each). Endpoints: composite structural heart disease (moderate or greater, EchoNext definition), LVEF $\le$45\,\%, LV wall thickness $\ge$13\,mm and reduced ejection fraction (LVEF $\le$40\,\% vs $\ge$50\,\%; post hoc).

\section{{NT-proBNP readout}}\label{{sec:ntp}}
\begin{{figure}}[p]\centering\includegraphics[width=\linewidth]{{harvi_fig2_ntprobnp.png}}
\caption{{NT-proBNP readout. (a--c) MIMIC-IV out-of-fold ROC curves for rule-out ($<$300\,pg/mL), age-adjusted rule-in and severe ($>$3{{,}}000\,pg/mL). (d, e) Emory, zero-shot: ICD-coded HFrEF and HFpEF against age- and sex-matched controls. (f) EchoNext held-out test patients, reduced EF$^\dagger$, zero-shot. Filled circles: Youden points with thresholds chosen on separate data (Section~\ref{{sec:ops}}); open circles: default (head probability $\geq$0.5) or transferred 300\,pg/mL cut-off; open diamonds: transferred age-adjusted rule-in cut-offs. (g) Estimated versus measured NT-proBNP per ECG, coloured by density (log axes; ECGs with estimates at the 10\,pg/mL floor not shown; dashed: line of identity). (h) Rule-out decision at the default operating point; positive class = measured NT-proBNP $<$300\,pg/mL. $^\dagger$LVEF $\le$40\,\% vs $\ge$50\,\%, post hoc.}}\label{{fig:ntp}}\end{{figure}}
\paragraph{{Development.}} On MIMIC-IV, HARVI-ECG reached AUROC {ci(ro)} for rule-out, {ci(ri)} for rule-in and {ci(sv)} for the severe decision; the mean of rule-out and rule-in was {ci(mo)} (Table~\ref{{tab:int}}, Figure~\ref{{fig:ntp}}a--c). The estimate ranks patients consistently with measured NT-proBNP (Spearman $\rho$ = {cal['rF']:.3f}; Figure~\ref{{fig:ntp}}g). At the default operating point, the rule-out decision has sensitivity {ro_sens:.3f} and specificity {ro_spec:.3f} (Figure~\ref{{fig:ntp}}h). {n(n_pre)} of the {n(n_pat)} patients had ECG waveforms in the pretraining corpus; on the {n(umo.n_pat)} patients without, the mean rule-out/rule-in AUROC was {ci(umo)} (Table~\ref{{tab:unexp}}).

{int_table(A, "MIMIC-IV out-of-fold results, all patients", "tab:int")}
{int_table(U, "MIMIC-IV out-of-fold results, patients whose ECGs were not in the pretraining corpus", "tab:unexp")}

\paragraph{{Unseen hospitals.}} Applied unchanged, the NT-proBNP estimate detected HFrEF at Emory with AUROC {ci(efr)} and HFpEF with {ci(efp)} (Table~\ref{{tab:ext}}, Figure~\ref{{fig:ntp}}d,e). At EchoNext it identified reduced ejection fraction with AUROC {x40.surrogate_auroc:.3f} and LVEF $\le$45\,\% with {x45.surrogate_auroc:.3f} (Figure~\ref{{fig:ntp}}f). How the fixed pg/mL cut-offs transfer is analysed in Section~\ref{{sec:ops}}.

\begin{{table}}[t]\centering\small\setlength{{\tabcolsep}}{{4pt}}
\caption{{Zero-shot results at Emory (score: NT-proBNP estimate, mean of the five fold models). Sensitivity and specificity at the fixed MIMIC-IV cut-offs (300\,pg/mL and age-adjusted rule-in).}}\label{{tab:ext}}
\begin{{tabular}}{{@{{}}llcccc@{{}}}}\toprule
Endpoint & Model & \makecell{{AUROC\\(95\,\% CI)}} & \makecell{{$\Delta$ vs\\baseline}} & \makecell{{Sens / spec\\at 300}} & \makecell{{Sens / spec\\at rule-in}} \\\midrule
{chr(10).join(ext_rows)}
\bottomrule\end{{tabular}}\end{{table}}

\section{{Comparison with a task-specific baseline}}\label{{sec:found}}
\begin{{figure}}[t]\centering\includegraphics[width=\linewidth]{{harvi_fig3_foundation.png}}
\caption{{HARVI-ECG against the task-specific baseline (trained from scratch on the same MIMIC-IV folds), on identical patients. AUROC with 95\,\% patient-bootstrap CI; right column: paired $\Delta$AUROC (95\,\% CI).}}\label{{fig:found}}\end{{figure}}
Under the pre-specified rule, HARVI-ECG improved the mean rule-out/rule-in AUROC over the task-specific baseline by {dci(mo)}, with the CI excluding zero (Figure~\ref{{fig:found}}, Table~\ref{{tab:int}}). The gain was largest at the unseen hospital: Emory HFrEF {dci(efr)} and HFpEF {dci(efp)}. The baseline could not be applied at EchoNext, whose waveforms are released per-lead standardised; HARVI-ECG normalises each lead itself and is unaffected.

\section{{SHD module}}\label{{sec:shd}}
\begin{{figure}}[t]\centering\includegraphics[width=\linewidth]{{harvi_fig4_shd.png}}
\caption{{SHD module on held-out EchoNext test patients. (a) ROC for reduced EF$^\dagger$: NT-proBNP readout (zero-shot) and SHD module (adapted on EchoNext); filled circles are Youden points chosen on the validation split. (b) AUROC (95\,\% CI) per endpoint; right column: paired $\Delta$. (c, d) AUROC of a frozen-encoder logistic-regression probe (regularisation chosen on the validation split) against the number of echo-labelled training patients (log scale); line: mean, band: min--max of 5 draws per size, except the full training set (26{{,}}218 patients), which is a single fit; dashed: NT-proBNP readout with no local labels. $^a$Moderate or greater echo finding. $^\dagger$Post hoc.}}\label{{fig:shd}}\end{{figure}}
Adapted on EchoNext echocardiography labels ({n(int(le('shd',26218).n_patients))} training patients; mean of three seeds), the SHD module outperformed the zero-shot NT-proBNP readout on every endpoint (Table~\ref{{tab:echo}}, Figure~\ref{{fig:shd}}a,b). It reached AUROC {x40.direct_auroc:.3f} for reduced ejection fraction, {x45.direct_auroc:.3f} for LVEF $\le$45\,\% ({o45.direct_auroc:.3f} in outpatients) and {xs.direct_auroc:.3f} for composite structural heart disease. The largest gain was for LV wall thickness ({dd(xw)}), the phenotype NT-proBNP reflects least.

Few labels are needed (Figure~\ref{{fig:shd}}c,d). A logistic-regression probe on the frozen encoder, with its regularisation chosen on the validation split, matched the zero-label NT-proBNP readout with {cross['lvef_le45']} labelled patients for LVEF $\le$45\,\% and LV wall thickness and with {n(cross['shd'])} for structural heart disease, and kept improving with more labels (LVEF $\le$45\,\%: {le('lvef_le45',26218)['mean']:.3f} with all training patients).

\begin{{table}}[t]\centering\small
\caption{{EchoNext held-out test patients. AUROC with 95\,\% patient-bootstrap CI and paired difference (SHD module $-$ NT-proBNP readout). $^a$Composite of moderate-or-greater echo findings. $^b$Post hoc endpoint.}}\label{{tab:echo}}
\setlength\tabcolsep{{4pt}}\footnotesize
\begin{{tabular}}{{lrccc}}\toprule
Endpoint & Cases / $n$ & \makecell{{NT-proBNP\\readout}} & \makecell{{SHD\\module}} & $\Delta$ \\\midrule
{XROWS}
\bottomrule\end{{tabular}}\end{{table}}

The two readouts are complementary: the NT-proBNP readout works at a new site without local labels, and a few hundred echo-labelled patients enable a dedicated structural heart disease screener.

\section{{Operating points}}\label{{sec:ops}}
\begin{{figure}}[t]\centering\includegraphics[width=0.92\linewidth]{{harvi_fig5_operating_points.png}}
\caption{{Sensitivity and specificity at operating points (95\,\% CI, patient bootstrap, threshold fixed). (a) MIMIC-IV: cross-fitted Youden threshold (filled) and default threshold (open). (b) Emory: MIMIC-IV cut-offs transferred unchanged (open circle: 300\,pg/mL; open diamond: age-adjusted rule-in) and a Youden threshold recalibrated at Emory (filled). (c) EchoNext test: Youden thresholds chosen on the validation split, for both readouts. Values are listed in Table~\ref{{tab:ops}} (Appendix~\ref{{app:ops}}). $^a$Moderate or greater echo finding. $^\dagger$Post hoc.}}\label{{fig:ops}}\end{{figure}}
Each threshold was chosen on data separate from the patients on which it is evaluated: in MIMIC-IV by cross-fitting (the Youden threshold of the other four folds, applied to the held-out fold), at Emory by splitting the patients into halves (Youden threshold from one half applied to the other; median of 20 repeats), and at EchoNext on the validation split. Confidence intervals come from 2{{,}}000 patient-bootstrap draws with the threshold held fixed.

In MIMIC-IV, the Youden thresholds give balanced sensitivity and specificity for all three decisions (rule-out {ss(op('rule_out_lt300', YD))}, rule-in {ss(op('rule_in_age_adjusted', YD))}, severe {ss(op('severe_gt3000', YD))}; Figure~\ref{{fig:ops}}a), whereas the default threshold favours sensitivity for rule-in and severe. At Emory, the age-adjusted rule-in cut-offs transferred from MIMIC-IV performed as well as a threshold recalibrated at Emory itself (HFrEF {ss(oe)} versus {ss(oep)}; Figure~\ref{{fig:ops}}b). The fixed 300\,pg/mL rule-out cut-off flagged most patients at Emory (specificity {p2(o3.specificity)} for HFrEF and {p2(o3p.specificity)} for HFpEF) and is the cut-off that needs local recalibration. On EchoNext, at validation-chosen thresholds, the SHD module reached sensitivity/specificity {ss(xsh)} for reduced ejection fraction, against {ss(xn)} for the NT-proBNP readout (Figure~\ref{{fig:ops}}c).

\section{{Comparison with published models}}\label{{sec:bench}}
\begin{{figure}}[h]\centering\includegraphics[width=\linewidth]{{harvi_benchmark.png}}
\caption{{HARVI-ECG against published ECG-to-NT-proBNP models. Published values as reported: Liu et al.~\cite{{liu2023}}, NT-proBNP $\geq$500\,pg/mL in their external community-hospital set (AUC 0.880; 0.879 at $\geq$1{{,}}000\,pg/mL; no CI reported); Lopez Alcaraz and Strodthoff~\cite{{lopezalcaraz2025}}, NT-proBNP $\geq$353\,pg/mL on a MIMIC-IV test fold from the ECG with demographics, biometrics and vital signs (0.903, 95\,\% CI 0.883--0.923). HARVI-ECG: MIMIC-IV out-of-fold, 95\,\% patient-bootstrap CI. Thresholds, cohorts and inputs differ between studies, so the values are not a head-to-head comparison; the axis starts at 0.6.}}\label{{fig:bench}}\end{{figure}}
HARVI-ECG is on par with published ECG-to-NT-proBNP models (Figure~\ref{{fig:bench}}). Its rule-out AUROC ({ci(ro)}; detection of NT-proBNP $\geq$300\,pg/mL) is comparable to the 0.880 reported by Liu et al.~\cite{{liu2023}} at their external hospital for $\geq$500\,pg/mL, and to the 0.903 reported by Lopez Alcaraz and Strodthoff~\cite{{lopezalcaraz2025}} for $\geq$353\,pg/mL in MIMIC-IV, which additionally used vital signs and biometrics. HARVI-ECG additionally provides zero-shot validation at two unseen hospitals and a structural heart disease readout.

\section{{Limitations}}\label{{sec:lim}}
\begin{{enumerate}}
\item Retrospective evaluation; prospective validation of the screening pathway is planned.
\item Emory phenotypes are ICD-10 based. EchoNext records echo findings but not symptoms, so its positives include symptomatic heart failure as well as asymptomatic structural disease (Stage B).
\item The 300\,pg/mL rule-out cut-off requires site recalibration (Section~\ref{{sec:ops}}). The NT-proBNP regression head was trained on the raw pg/mL scale (the pre-specification described a log scale); a log-scale head is planned. The decision rule uses the classification outputs and is unaffected.
\item Each fold model of the NT-proBNP readout was trained with a single seed.
\end{{enumerate}}

\section{{Availability}}\label{{sec:avail}}
Analysis code, aggregate results and figures are available in the accompanying repository and Zenodo record. Model weights (encoder, NT-proBNP readout and SHD module) are available upon reasonable request through a gated Hugging Face repository, \url{{https://huggingface.co/maxhaberbusch/HARVI-ECG}}; access requests are reviewed individually. The weights are licensed under the HARVI-ECG Model Weights Research Licence (non-commercial research only, no clinical use, no redistribution); HARVI-ECG is not a medical device. MIMIC-IV and HEEDB are credentialed-access datasets and EchoNext is available under its data-use agreement; patient-level data and predictions are not redistributed.

\begin{{thebibliography}}{{9}}\small\setlength{{\itemsep}}{{1pt}}
\bibitem{{gow2023}} B.~Gow, T.~Pollard, L.~A. Nathanson, et~al. MIMIC-IV-ECG: diagnostic electrocardiogram matched subset (version 1.0). \emph{{PhysioNet}}, 2023. doi:10.13026/4nqg-sb35.
\bibitem{{johnson2023}} A.~E.~W. Johnson, L.~Bulgarelli, L.~Shen, et~al. MIMIC-IV, a freely accessible electronic health record dataset. \emph{{Scientific Data}}, 10:1, 2023.
\bibitem{{koscova2026}} Z.~Koscova, Q.~Li, C.~Robichaux, et~al. The Harvard-Emory ECG Database. \emph{{Scientific Data}}, 13:516, 2026. doi:10.1038/s41597-026-06861-9.
\bibitem{{beck2024}} M.~Beck, K.~P\"oppel, M.~Spanring, et~al. xLSTM: Extended long short-term memory. In \emph{{Advances in Neural Information Processing Systems (NeurIPS)}}, 2024. arXiv:2405.04517.
\bibitem{{liu2023}} P.-Y. Liu, C.~Lin, C.-S. Lin, W.-H. Fang, C.-C. Lee, C.-H. Wang, and D.-J. Tsai. Artificial intelligence-enabled electrocardiography detects B-type natriuretic peptide and N-terminal pro-brain natriuretic peptide. \emph{{Diagnostics}}, 13(17):2723, 2023. doi:10.3390/diagnostics13172723.
\bibitem{{lopezalcaraz2025}} J.~M. Lopez Alcaraz and N.~Strodthoff. Abnormality prediction and forecasting of laboratory values from electrocardiogram signals using multimodal deep learning. \emph{{Scientific Reports}}, 15:39362, 2025. doi:10.1038/s41598-025-26715-5.
\bibitem{{hu2022}} E.~J. Hu, Y.~Shen, P.~Wallis, et~al. LoRA: Low-rank adaptation of large language models. In \emph{{International Conference on Learning Representations (ICLR)}}, 2022. arXiv:2106.09685.
\end{{thebibliography}}

\appendix
\section{{Operating points}}\label{{app:ops}}
\begin{{table}}[h]\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\caption{{Operating points (values shown in Figure~\ref{{fig:ops}}). Sensitivity and specificity with 95\,\% CI (2{{,}}000 patient-bootstrap draws, threshold fixed). MIMIC-IV probability thresholds are the median of the five fold thresholds. For rule-out, the positive class is measured NT-proBNP $<$300\,pg/mL. $^a$Moderate or greater echo finding. $^\dagger$LVEF $\le$40\,\% vs $\ge$50\,\%, post hoc.}}\label{{tab:ops}}
\begin{{tabular}}{{@{{}}llcc@{{}}}}\toprule
Operating point & Threshold & Sensitivity & Specificity \\\midrule
{OPROWS}
\bottomrule\end{{tabular}}\end{{table}}
\end{{document}}
"""
assert mo["head"] == MO and ro["head"] == "rule-out" and ri["head"] == "rule-in (age-adj.)" and sv["head"] == "severe"
assert abs(mo.auroc - 0.894) < 5e-4 and abs(ro.auroc - 0.920) < 5e-4 and sv.d_vs_harvi < 0 and abs(hmo.auroc - 0.882) < 5e-4
bad = sorted({c for c in T if ord(c) > 127}); assert not bad, f"non-ASCII characters for pdflatex: {bad}"

os.makedirs("report", exist_ok=True)
open("report/harvi_results_report.tex", "w").write(T)
