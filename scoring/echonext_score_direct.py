"""Pre-specified EchoNext comparison (echonext_preregistration.md §3-4): direct FairECG classifier (D) vs NT-proBNP surrogate (S),
combination (C), label-efficiency crossover. Test split, one ECG per patient; patient bootstrap 2,000 draws, seed 0.
Outputs: echonext_direct_vs_surrogate.csv, echonext_label_efficiency_summary.csv
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
DATA = os.environ.get("ECHONEXT_DATA", "data")
OUTD = os.environ.get("ECHONEXT_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "echonext")); DP = f"{DATA}/direct/preds"; B = 2000
S = pd.read_parquet(f"{DATA}/surrogate_scores.parquet")
EPS = {"shd": ("shd", "SHD (composite)"), "lvef45": ("lvef_le45", "LVEF <=45%"), "lvwt13": ("lvwt_ge13", "LV wall >=13 mm")}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def direct(ep, split):
    k, ky = ("uid", "lora") if split == "test" else ("val_uid", "val_lora")
    zs = [np.load(f"{DP}/fairlora_preds_fairecgv2clin_echonext_{ep}_erm_s{s}.npz") for s in (0, 1, 2)]
    for z in zs[1:]: assert (z[k] == zs[0][k]).all()
    return pd.DataFrame({"ecg_uid": zs[0][k], f"D_{ep}": np.mean([z[ky] for z in zs], 0), **{f"D_{ep}_s{s}": z[ky] for s, z in enumerate(zs)}})
T, V = S[S.split == "test"].copy(), S[S.split == "val"].copy()
for ep in EPS:
    T = T.merge(direct(ep, "test"), on="ecg_uid", how="left", validate="1:1"); V = V.merge(direct(ep, "val"), on="ecg_uid", how="left", validate="1:1")
assert T.filter(like="D_").notna().all().all() and V.filter(like="D_").notna().all().all() and len(T) == 5442 and len(V) == 4626
T = T.sort_values("ecg_uid").reset_index(drop=True)
BOOT = np.random.default_rng(0).integers(0, len(T), size=(B, len(T)))

def paired(y, a, b, m):
    ea, eb = roc_auc_score(y[m], a[m]), roc_auc_score(y[m], b[m]); va, vd = [], []
    for idx in BOOT:
        k = idx[m[idx]]; yk = y[k]
        if 0 < yk.sum() < len(yk):
            x, z = roc_auc_score(yk, a[k]), roc_auc_score(yk, b[k]); va.append(x); vd.append(x - z)
    return ea, *np.percentile(va, [2.5, 97.5]), eb, ea - eb, *np.percentile(vd, [2.5, 97.5])
def ci(y, a, m):
    e = roc_auc_score(y[m], a[m]); v = [roc_auc_score(y[i], a[i]) for i in (idx[m[idx]] for idx in BOOT) if 0 < y[i].sum() < len(i)]
    return e, *np.percentile(v, [2.5, 97.5])

rows = []
endpoints = [(ep, col, nm, None) for ep, (col, nm) in EPS.items()] + [("lvef45", "lvef_value", "LVEF <=40% vs >=50% (post hoc)", "ef40")]
for ep, col, nm, special in endpoints:
    if special == "ef40":
        lv = T.lvef_value.values; y = (lv <= 40).astype(float); keep = (lv <= 40) | (lv >= 50)
        lvv = V.lvef_value.values; yv = (lvv <= 40).astype(float); keepv = (lvv <= 40) | (lvv >= 50)
    else:
        y = T[col].values.astype(float); keep = np.ones(len(T), bool); yv = V[col].values.astype(float); keepv = np.ones(len(V), bool)
    Dv, Sv = T[f"D_{ep}"].values, T.mu_mean.values
    # combination: fit on val (logit D, standardised S), score test
    mu_m, mu_s = V.mu_mean[keepv].mean(), V.mu_mean[keepv].std()
    Xv = np.c_[logit(V[f"D_{ep}"].values[keepv]), (V.mu_mean.values[keepv] - mu_m) / mu_s]
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xv, yv[keepv])
    Cv = lr.decision_function(np.c_[logit(Dv), (Sv - mu_m) / mu_s])
    seeds = [roc_auc_score(y[keep], T[f"D_{ep}_s{s}"].values[keep]) for s in (0, 1, 2)]
    for sub, sm in (("all test", np.ones(len(T), bool)), ("outpatient", (T.care_setting == "outpatient").values)):
        m = keep & sm
        dA, dlo, dhi, sA, dd, ddlo, ddhi = paired(y, Dv, Sv, m)
        sA_, slo, shi = ci(y, Sv, m)
        cA, clo, chi, _, cd, cdlo, cdhi = paired(y, Cv, Dv, m)
        rows.append(dict(subset=sub, endpoint=nm, n=int(m.sum()), n_pos=int(y[m].sum()),
                         direct_auroc=dA, direct_lo=dlo, direct_hi=dhi, surrogate_auroc=sA_, surrogate_lo=slo, surrogate_hi=shi,
                         d_direct_minus_surrogate=dd, d_lo=ddlo, d_hi=ddhi,
                         combined_auroc=cA, combined_lo=clo, combined_hi=chi, d_combined_minus_direct=cd, dc_lo=cdlo, dc_hi=cdhi,
                         direct_seed_min=min(seeds) if sub == "all test" else np.nan, direct_seed_max=max(seeds) if sub == "all test" else np.nan,
                         combo_coef_D=lr.coef_[0][0], combo_coef_S=lr.coef_[0][1]))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUTD, "echonext_direct_vs_surrogate.csv"), index=False)

# label efficiency: mean frozen-probe AUROC per n vs surrogate; LoRA subset cells; age+sex baseline
LE = pd.read_csv(f"{DATA}/direct/label_efficiency.csv")
sur = {c: R[(R.subset == "all test") & (R.endpoint == nm)].surrogate_auroc.iloc[0] for _, (c, nm) in EPS.items()}
L = LE.groupby(["endpoint", "n_patients"]).test_auroc.agg(["mean", "min", "max", "count"]).reset_index()
L["surrogate_auroc"] = L.endpoint.map(sur); L["probe_ge_surrogate"] = L["mean"] >= L.surrogate_auroc
for p, n in (("p1000", 1000), ("p5000", 5000)):
    z = np.load(f"{DP}/fairlora_preds_fairecgv2clin_echonext_shd_{p}_erm_s0.npz")
    L.loc[(L.endpoint == "shd") & (L.n_patients == n), "lora_subset_auroc"] = roc_auc_score(z["y"], z["lora"])
L.to_csv(os.path.join(OUTD, "echonext_label_efficiency_summary.csv"), index=False)
X = {c: L[(L.endpoint == c) & L.probe_ge_surrogate].n_patients.min() for c in sur}
print(R[["subset", "endpoint", "n", "n_pos", "direct_auroc", "surrogate_auroc", "d_direct_minus_surrogate", "d_lo", "d_hi", "combined_auroc", "d_combined_minus_direct", "dc_lo", "dc_hi"]].round(3).to_string())
print("crossover n (smallest n with mean frozen probe >= surrogate):", X)
print(L.round(3).to_string())
