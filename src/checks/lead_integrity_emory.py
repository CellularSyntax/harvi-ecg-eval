import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
import numpy as np, pandas as pd, json, wfdb, collections
import ecg_audit.data.waveforms as W
from ecg_audit.data.locators import build_resolver
HE = f"{_PR}"
em = pd.read_parquet(f"{HE}/manifests/manifest_emory.parquet")
sub = em.sample(300, random_state=0).reset_index(drop=True)
res = build_resolver("emory", f"{HE}/data", sub)
names = collections.Counter(); R = {"n": 0, "avl_eq_avf": 0, "res": collections.defaultdict(list), "fs": collections.Counter(), "len": collections.Counter()}
for i in range(len(sub)):
    try:
        loc = res(sub.iloc[i]); rec = wfdb.rdheader(loc["record_path"])
        names[tuple(rec.sig_name)] += 1; R["fs"][int(rec.fs)] += 1; R["len"][int(rec.sig_len)] += 1
        s = W.load_canonical("emory", loc).astype(np.float64)
    except Exception as e:
        R.setdefault("err", str(e)[:120]); continue
    R["n"] += 1
    I, II, III, aVR, aVL, aVF = s[:6]
    nr = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))
    if np.allclose(aVL, aVF): R["avl_eq_avf"] += 1
    A = np.stack([I, II], 1)
    for nm, ch in (("aVL", aVL), ("aVF", aVF)):
        co, *_ = np.linalg.lstsq(A, ch, rcond=None); R.setdefault("coef_"+nm, []).append(co.tolist())
        R.setdefault("fitres_"+nm, []).append(float(np.linalg.norm(A @ co - ch) / (np.linalg.norm(ch) + 1e-9)))
    R["res"]["III=II-I"].append(nr(III, II - I)); R["res"]["aVR"].append(nr(aVR, -(I + II) / 2))
    R["res"]["aVL"].append(nr(aVL, I - II / 2)); R["res"]["aVF"].append(nr(aVF, II - I / 2))
out = {"n": R["n"], "avl_eq_avf": R["avl_eq_avf"], "median_resid": {k: float(np.median(v)) for k, v in R["res"].items()},
       "frac_aVF_resid_gt_0.2": float(np.mean(np.array(R["res"]["aVF"]) > 0.2)), "frac_aVL_resid_gt_0.2": float(np.mean(np.array(R["res"]["aVL"]) > 0.2)),
       "sig_name_sets": {"|".join(k): v for k, v in names.most_common(6)}, "fs": dict(R["fs"]), "len": dict(R["len"].most_common(4)), "err": R.get("err"),
       "coef_aVL_median": np.median(np.array(R["coef_aVL"]), 0).tolist(), "coef_aVF_median": np.median(np.array(R["coef_aVF"]), 0).tolist(),
       "coef_aVL_iqr": np.percentile(np.array(R["coef_aVL"]), [25, 75], 0).tolist(), "coef_aVF_iqr": np.percentile(np.array(R["coef_aVF"]), [25, 75], 0).tolist(),
       "fitres_aVL_median": float(np.median(R["fitres_aVL"])), "fitres_aVF_median": float(np.median(R["fitres_aVF"]))}
print(json.dumps(out, indent=1))
