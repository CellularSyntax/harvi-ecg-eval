"""Pre-specified scoring of the NT-proBNP surrogate (score S) on EchoNext test (echonext_preregistration.md §2-4, Amendment 1).
Usage: ECHONEXT_DATA=/path/to/exports python scoring/echonext_score_surrogate.py
Input: surrogate_scores.parquet (val + test, one ECG per patient in each). Output: echonext_surrogate_auroc.csv, echonext_surrogate_thresholds.csv
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
DATA = os.environ.get("ECHONEXT_DATA", "data"); B = 2000
OUTD = os.environ.get("ECHONEXT_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "echonext"))
S = pd.read_parquet(f"{DATA}/surrogate_scores.parquet")
assert S.ecg_uid.is_unique and set(S.split) == {"val", "test"}
for sp in ("val", "test"): assert S[S.split == sp].patient_id.is_unique   # one ECG per patient -> row bootstrap == patient bootstrap

def endpoints(d):
    lv = d.lvef_value
    return {"SHD (composite)": (d.shd.astype(float), np.ones(len(d), bool)),
            "LVEF <=45%": (d.lvef_le45.astype(float), np.ones(len(d), bool)),
            "LV wall >=13 mm": (d.lvwt_ge13.astype(float), np.ones(len(d), bool)),
            "LVEF <=40% vs >=50% (post hoc)": ((lv <= 40).astype(float), (lv <= 40) | (lv >= 50))}
def thr_in(age): return np.where(age < 50, 450., np.where(age > 75, 1800., 900.))
SUBSETS = {"all test": lambda d: np.ones(len(d), bool), "outpatient": lambda d: (d.care_setting == "outpatient").values}
rng = np.random.default_rng(0)
te = S[S.split == "test"].reset_index(drop=True); va = S[S.split == "val"].reset_index(drop=True)
BOOT = rng.integers(0, len(te), size=(B, len(te)))

def ci_auc(y, s, m):
    yy, ss = y[m], s[m]; est = roc_auc_score(yy, ss); v = []
    for idx in BOOT:
        k = idx[m[idx]]; yk = y[k]
        if 0 < yk.sum() < len(yk): v.append(roc_auc_score(yk, s[k]))
    return est, *np.percentile(v, [2.5, 97.5])

def ci_ops(y, flag, m):
    def ops(yv, fv):
        tp = (fv & (yv == 1)).sum(); fn = (~fv & (yv == 1)).sum(); tn = (~fv & (yv == 0)).sum(); fp = (fv & (yv == 0)).sum()
        return np.array([tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1), tn / max(tn + fn, 1), fv.mean()])
    est = ops(y[m], flag[m]); v = np.array([ops(y[idx[m[idx]]], flag[idx[m[idx]]]) for idx in BOOT])
    lo, hi = np.nanpercentile(v, [2.5, 97.5], axis=0); return est, lo, hi

A, T = [], []
for sub, sf in SUBSETS.items():
    base = sf(te)
    for ep, (y, keep) in endpoints(te).items():
        y = y.values; m = base & np.asarray(keep) & ~np.isnan(y)
        for sc, s in (("predicted NT-proBNP", te.mu_mean.values), ("rule-in probability", te.p_rulein_mean.values), ("age only (context, not pre-specified)", te.age.values.astype(float))):
            e, lo, hi = ci_auc(y, s, m)
            A.append(dict(subset=sub, endpoint=ep, score=sc, n=int(m.sum()), n_pos=int(y[m].sum()), prevalence=float(y[m].mean()), auroc=e, lo=lo, hi=hi))
        # thresholds: fixed MIMIC cut-offs + val-recalibrated cut-offs (chosen on val, same endpoint / subset definition)
        yv, kv = endpoints(va)[ep]; yv = yv.values; mv = sf(va) & np.asarray(kv) & ~np.isnan(yv); sv = va.mu_mean.values[mv]; yvm = yv[mv]
        t_sens = np.quantile(sv[yvm == 1], 0.10)          # >= 90 % sensitivity on val
        t_spec = np.quantile(sv[yvm == 0], 0.90)          # >= 90 % specificity on val
        mu = te.mu_mean.values
        rules = {"fixed: predicted >=300 pg/mL": mu >= 300, "fixed: predicted > age-adjusted rule-in cut": mu > thr_in(te.age.values),
                 "fixed: rule-in probability >=0.5": te.p_rulein_mean.values >= 0.5,
                 f"val-recalibrated, 90% sensitivity (>= {t_sens:.0f} pg/mL)": mu >= t_sens,
                 f"val-recalibrated, 90% specificity (>= {t_spec:.0f} pg/mL)": mu >= t_spec}
        for rn, fl in rules.items():
            e, lo, hi = ci_ops(y, fl, m)
            T.append(dict(subset=sub, endpoint=ep, rule=rn, n=int(m.sum()), prevalence=float(y[m].mean()),
                          sens=e[0], sens_lo=lo[0], sens_hi=hi[0], spec=e[1], spec_lo=lo[1], spec_hi=hi[1],
                          ppv=e[2], ppv_lo=lo[2], ppv_hi=hi[2], npv=e[3], npv_lo=lo[3], npv_hi=hi[3], flagged=e[4]))
pd.DataFrame(A).to_csv(os.path.join(OUTD, "echonext_surrogate_auroc.csv"), index=False); pd.DataFrame(T).to_csv(os.path.join(OUTD, "echonext_surrogate_thresholds.csv"), index=False)
print("written", len(A), len(T))
