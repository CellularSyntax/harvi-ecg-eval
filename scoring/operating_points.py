"""Operating points for the ROC curves of the final HARVI-ECG figures.
No threshold is chosen on the rows where its sensitivity/specificity are reported:
  MIMIC-IV  cross-fitted Youden: threshold from the OOF predictions of the other 4 folds, applied to the held-out fold;
            plus the default operating point (head probability >= 0.5).
  Emory     transferred MIMIC cut-offs on the NT-proBNP estimate (300 pg/mL; age-adjusted rule-in), plus a
            site-recalibrated Youden point, 2-fold cross-fitted within Emory (patient-level split, 20 repeats; the
            reported point is the median over repeats).
  EchoNext  Youden threshold chosen on the validation split, applied to the test split (both readouts).
CIs: 2,000 patient-bootstrap draws with the threshold held fixed.  Output: operating_points.csv / .json"""
import glob, json, os, numpy as np, pandas as pd
DATA = os.environ.get("HARVI_DATA", "data")            # credentialed exports (harvi_oof_merged.csv, score_export.npz, score_emory_meta.csv)
EDATA = os.environ.get("ECHONEXT_DATA", DATA)          # EchoNext exports (surrogate_scores.parquet, direct/preds/)
OUT = os.environ.get("HARVI_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))

rng0 = np.random.default_rng(0); B = 2000
D = pd.read_csv(f"{DATA}/harvi_oof_merged.csv"); Z = np.load(f"{DATA}/score_export.npz")
EM = pd.read_csv(f"{DATA}/score_emory_meta.csv"); S = pd.read_parquet(f"{EDATA}/surrogate_scores.parquet")

def roc(y, s):
    o = np.argsort(-s, kind="mergesort"); ys, ss = y[o], s[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); keep = np.r_[np.diff(ss) != 0, True]
    tpr = np.r_[0, tp[keep] / ys.sum()]; fpr = np.r_[0, fp[keep] / (len(ys) - ys.sum())]; thr = np.r_[np.inf, ss[keep]]
    return fpr, tpr, thr
def auc(y, s):
    f, t, _ = roc(y, s); return float(np.trapezoid(t, f))
def youden_thr(y, s):
    f, t, th = roc(y, s); k = int(np.argmax(t - f)); return float(th[k])
def ops(y, flag):
    tp = (flag & (y == 1)).sum(); fn = ((~flag) & (y == 1)).sum(); tn = ((~flag) & (y == 0)).sum(); fp = (flag & (y == 0)).sum()
    return np.array([tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1), tn / max(tn + fn, 1), flag.mean()])
def boot(y, flag, groups, b=B, seed=0):
    rng = np.random.default_rng(seed); ug, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(ug))] if len(ug) < len(y) else None
    out = []
    for _ in range(b):
        if idx_by is None: ii = rng.integers(0, len(y), len(y))
        else: ii = np.concatenate([idx_by[k] for k in rng.integers(0, len(ug), len(ug))])
        out.append(ops(y[ii], flag[ii]))
    out = np.array(out); return np.nanpercentile(out, 2.5, 0), np.nanpercentile(out, 97.5, 0)

ROWS = []
def add(fig, panel, cohort, series, kind, method, y, flag, groups, thr, thr_unit, s_for_point):
    e = ops(y, flag); lo, hi = boot(y, flag, groups)
    ROWS.append(dict(figure=fig, panel=panel, cohort=cohort, series=series, point=kind, threshold_method=method,
                     threshold=thr, threshold_unit=thr_unit, n=int(len(y)), n_pos=int(y.sum()),
                     fpr=1 - e[1], tpr=e[0], sensitivity=e[0], sens_lo=lo[0], sens_hi=hi[0], specificity=e[1], spec_lo=lo[1], spec_hi=hi[1],
                     ppv=e[2], npv=e[3], flagged=e[4], auroc_check=auc(y, s_for_point)))

# ---------- MIMIC-IV, NT-proBNP readout (HARVI-ECG = FairECG LoRA heads)
y_nt = D.ntprobnp.values; age = D.age.values; fold = D.fold.values; pid = D.subject_id.values
thr_in = np.where(age < 50, 450, np.where(age > 75, 1800, 900))
LAB = {"rule_out_lt300": (y_nt < 300).astype(int), "rule_in_age_adjusted": (y_nt > thr_in).astype(int), "severe_gt3000": (y_nt > 3000).astype(int)}
P = Z["fairecgv2clin_lora_p"]
for h, (pan, y) in enumerate(LAB.items()):
    s = P[:, h]; assert abs(auc(y, s) - {"rule_out_lt300": .920, "rule_in_age_adjusted": .868, "severe_gt3000": .848}[pan]) < 6e-4, (pan, auc(y, s))
    flag = np.zeros(len(y), bool); thrs = []
    for k in np.unique(fold):
        t = youden_thr(y[fold != k], s[fold != k]); thrs.append(t); flag[fold == k] = s[fold == k] >= t
    add("harvi_fig2_ntprobnp", pan, "MIMIC-IV OOF", "HARVI-ECG", "Youden (cross-fitted)", "Youden threshold from the other 4 folds, applied to the held-out fold",
        y, flag, pid, float(np.median(thrs)), "head probability (median of 5 fold thresholds)", s)
    add("harvi_fig2_ntprobnp", pan, "MIMIC-IV OOF", "HARVI-ECG", "default", "head probability >= 0.5", y, s >= 0.5, pid, 0.5, "head probability", s)

# ---------- Emory, zero-shot (score = NT-proBNP estimate, 5-fold mean)
mu = Z["fairecgv2clin_lora_ext_mu"]; eage = EM.age.values; ethr_in = np.where(eage < 50, 450, np.where(eage > 75, 1800, 900))
for lab, pan in (("hfref", "hfref_vs_matched_controls"), ("hfpef", "hfpef_vs_matched_controls")):
    m = (EM.label.values == lab) | ((EM.label.values == "control") & (EM.ctl_for.values == lab))
    y = (EM.label.values[m] == lab).astype(int); s = mu[m]; g = EM.patient.values[m]
    assert abs(auc(y, s) - {"hfref": .810, "hfpef": .727}[lab]) < 6e-4, (lab, auc(y, s))
    add("harvi_fig2_ntprobnp", pan, "Emory zero-shot", "HARVI-ECG", "transferred cut-off 300 pg/mL", "MIMIC-IV rule-out cut-off, unchanged", y, s >= 300, g, 300.0, "pg/mL", s)
    add("harvi_fig2_ntprobnp", pan, "Emory zero-shot", "HARVI-ECG", "transferred age-adjusted rule-in", "MIMIC-IV age-adjusted rule-in cut-offs, unchanged", y, s > ethr_in[m], g, None, "450/900/1,800 pg/mL by age", s)
    # site-recalibrated: 2-fold cross-fitted Youden within Emory, 20 repeats -> pick the repeat with the median threshold
    reps = []
    for r in range(20):
        half = np.random.default_rng(100 + r).permutation(len(y)) % 2; flag = np.zeros(len(y), bool); th = []
        for k in (0, 1):
            t = youden_thr(y[half != k], s[half != k]); th.append(t); flag[half == k] = s[half == k] >= t
        reps.append((float(np.mean(th)), flag))
    reps.sort(key=lambda z: z[0]); thm, flag = reps[len(reps) // 2]
    add("harvi_fig2_ntprobnp", pan, "Emory zero-shot", "HARVI-ECG", "site-recalibrated Youden (cross-fitted)",
        "Youden threshold chosen on one half of the Emory patients, applied to the other half (2-fold, median of 20 repeats)", y, flag, g, thm, "pg/mL", s)

# ---------- EchoNext: NT-proBNP readout (zero-shot) and SHD module (adapted); Youden on val -> test
V, T = S[S.split == "val"], S[S.split == "test"]
def echo_y(df, ep):
    if ep == "ef40":
        lv = df.lvef_value.values; keep = (lv <= 40) | (lv >= 50); return (lv <= 40).astype(int), keep
    return df[ep].values.astype(int), np.ones(len(df), bool)
EPN = {"ef40": ("lvef_le40_vs_ge50", "lvef45"), "lvef_le45": ("lvef_le45", "lvef45"), "shd": ("shd", "shd"), "lvwt_ge13": ("lvwt_ge13", "lvwt13")}
REF = {("ef40", "nt"): .834, ("lvef_le45", "nt"): .830, ("shd", "nt"): .808, ("lvwt_ge13", "nt"): .685,
       ("ef40", "shd"): .929, ("lvef_le45", "shd"): .918, ("shd", "shd"): .850, ("lvwt_ge13", "shd"): .785}
def direct(model_ep):
    fs = sorted(glob.glob(f"{EDATA}/direct/preds/fairlora_preds_fairecgv2clin_echonext_{model_ep}_erm_s[012].npz")); assert len(fs) == 3, fs
    z = [np.load(f, allow_pickle=True) for f in fs]
    for q in z[1:]: assert np.array_equal(q["uid"], z[0]["uid"]) and np.array_equal(q["val_uid"], z[0]["val_uid"])
    te = pd.Series(np.mean([q["lora"] for q in z], 0), index=z[0]["uid"].astype(str)); va = pd.Series(np.mean([q["val_lora"] for q in z], 0), index=z[0]["val_uid"].astype(str))
    return va, te
for ep, (pan, model_ep) in EPN.items():
    yv, kv = echo_y(V, ep); yt, kt = echo_y(T, ep)
    sv, st = V.mu_mean.values, T.mu_mean.values
    assert abs(auc(yt[kt], st[kt]) - REF[(ep, "nt")]) < 6e-4, (ep, auc(yt[kt], st[kt]))
    t = youden_thr(yv[kv], sv[kv])
    add("harvi_fig4_shd" if ep == "ef40" else "echonext_other", pan, "EchoNext test", "HARVI-ECG · NT-proBNP readout (zero-shot)", "Youden (validation split)",
        "Youden threshold chosen on the EchoNext validation split, applied to the test split", yt[kt], st[kt] >= t, T.patient_id.values[kt], t, "pg/mL", st[kt])
    va, te = direct(model_ep)
    dv = va.reindex(V.ecg_uid.astype(str)).values; dt = te.reindex(T.ecg_uid.astype(str)).values
    assert np.isfinite(dt[kt]).all() and np.isfinite(dv[kv]).all()
    assert abs(auc(yt[kt], dt[kt]) - REF[(ep, "shd")]) < 6e-4, (ep, "shd", auc(yt[kt], dt[kt]))
    t = youden_thr(yv[kv], dv[kv])
    add("harvi_fig4_shd" if ep == "ef40" else "echonext_other", pan, "EchoNext test", "HARVI-ECG · SHD module (adapted on EchoNext)", "Youden (validation split)",
        "Youden threshold chosen on the EchoNext validation split, applied to the test split", yt[kt], dt[kt] >= t, T.patient_id.values[kt], t, "probability", dt[kt])

O = pd.DataFrame(ROWS)
num = O.select_dtypes("number").columns; O[num] = O[num].round(4)
O.to_csv(f"{OUT}/operating_points.csv", index=False); print(O[["panel", "series", "point", "threshold", "sensitivity", "specificity", "auroc_check"]].to_string())
