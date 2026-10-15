import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
"""Phase 0 inventory for the HARVI numbers (read-only except results/harvi/).
1) HARVI pairs + GroupKFold folds (must reproduce the training logs)
2) overlap of HARVI patients with FairECG's MIMIC pretraining patients
3) HF phenotype codes for HARVI patients (internal reference only)
4) candidate external cohorts: columns, sizes, HF / EF / BNP fields
"""
import numpy as np, pandas as pd, glob, os, json, gzip, collections
from sklearn.model_selection import GroupKFold
HE = f"{_PR}"; H = f"{_HR}"
OUT = f"{HE}/results/harvi"; os.makedirs(OUT, exist_ok=True)
R = {}

# ---- 1. HARVI pairs and folds -------------------------------------------------
m = np.load(f"{H}/cache/ntprobnp/metadata.npz", allow_pickle=True)
R["harvi_meta_keys"] = {k: list(m[k].shape) for k in m.files}
y = m["labels"].reshape(-1).astype(float); sid = m["ids"].astype(np.int64)
folds = np.full(len(y), -1)
for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups=sid)): folds[va] = k
R["harvi_n_pairs"] = int(len(y)); R["harvi_n_patients"] = int(len(np.unique(sid)))
R["harvi_fold_val_sizes"] = [int((folds == k).sum()) for k in range(5)]
R["harvi_label_quantiles"] = {q: float(np.quantile(y, q)) for q in (0, .05, .25, .5, .75, .95, 1)}
R["harvi_label_lt300"] = float((y < 300).mean())
R["harvi_flags_values"] = {str(k): int(v) for k, v in collections.Counter(map(str, m["flags"].reshape(len(y), -1)[:, 0])).items()} if "flags" in m.files else None
_p=f"{H}/cache/ntprobnp/ecg_data.npy"; _dt="float32" if abs(os.path.getsize(_p)-len(y)*12*5000*4)<1024 else "float64"
ecg = np.memmap(_p, mode="r", dtype=_dt, shape=(len(y),12,5000))   # identical to harvi_v2 load_data_memmap_info
R["harvi_ecg_shape"] = list(ecg.shape); R["harvi_ecg_dtype"] = str(ecg.dtype)
x0 = np.asarray(ecg[:64], dtype=np.float32)
R["harvi_ecg_amp"] = {"abs_p99": float(np.quantile(np.abs(x0), .99)), "mean": float(x0.mean()), "std": float(x0.std())}
np.savez(f"{OUT}/harvi_pairs_folds.npz", subject_id=sid, fold=folds, ntprobnp=y,
         age=m["age"].reshape(-1), gender=m["gender"].reshape(-1),
         delta_t=(m["delta_t"].reshape(-1) if "delta_t" in m.files else np.full(len(y), np.nan)))

# ---- 2. overlap with FairECG pretraining (MIMIC: all patients except equity val/test) ----
mf = pd.read_parquet(f"{HE}/manifests/manifest_mimic_full.parquet")
R["mimic_full_cols"] = list(mf.columns)[:60]; R["mimic_full_n"] = int(len(mf))
scol = next((c for c in mf.columns if c.lower() in ("split", "fold", "partition", "set")), None)
pcol = next((c for c in mf.columns if c.lower() in ("subject_id", "patient_id", "pid")), None)
R["mimic_split_col"] = scol; R["mimic_pid_col"] = pcol
if scol and pcol:
    R["mimic_split_values"] = {str(k): int(v) for k, v in mf[scol].value_counts().items()}
    heldout = set(mf.loc[mf[scol].astype(str).str.lower().isin(["val", "valid", "validation", "test"]), pcol].astype(np.int64))
    hs = set(np.unique(sid))
    R["harvi_patients_in_equity_valtest"] = int(len(hs & heldout))
    R["harvi_patients_in_pretraining"] = int(len(hs - heldout))
    np.save(f"{OUT}/harvi_patient_is_heldout.npy", np.isin(sid, list(heldout)))

# ---- 3. HF phenotype ICD codes for HARVI patients (internal reference) ----
hs = set(np.unique(sid).tolist()); ph = collections.defaultdict(set)
with gzip.open(f"{HE}/data/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz", "rt") as fh:
    for ch in pd.read_csv(fh, usecols=["subject_id", "icd_code", "icd_version"], dtype={"icd_code": str}, chunksize=2_000_000):
        ch = ch[ch.subject_id.isin(hs)]
        c = ch.icd_code.str.strip(); v = ch.icd_version
        for lab, mask in (("hfref", ((v == 10) & c.str.startswith("I502")) | ((v == 9) & c.str.startswith("4282"))),
                          ("hfpef", ((v == 10) & c.str.startswith("I503")) | ((v == 9) & c.str.startswith("4283"))),
                          ("hf_mixed", ((v == 10) & c.str.startswith("I504")) | ((v == 9) & c.str.startswith("4284"))),
                          ("hf_any", ((v == 10) & c.str.startswith("I50")) | ((v == 9) & c.str.startswith("428")))):
            ph[lab] |= set(ch.subject_id[mask])
R["harvi_patients_by_hf_code"] = {k: len(v) for k, v in ph.items()}

# ---- 4. external candidates ----------------------------------------------------
KEY = ("hf", "heart_fail", "ef", "lvef", "bnp", "echo", "icd", "i50", "428", "systolic", "diastolic", "split")
for name in ("emory", "echonext", "edbwh", "lead", "code15", "ikem"):
    p = f"{HE}/manifests/manifest_{name}.parquet"
    if not os.path.exists(p): continue
    d = pd.read_parquet(p); cols = [c for c in d.columns if any(k in c.lower() for k in KEY)]
    info = {"n": int(len(d)), "n_cols": int(d.shape[1]), "key_cols": cols}
    for c in cols[:25]:
        s = d[c]
        if s.dtype.kind in "biuf" and s.nunique() > 12:
            info[c] = {"nonnull": int(s.notna().sum()), "q": [float(s.quantile(q)) for q in (.05, .5, .95)]}
        else:
            info[c] = {str(k): int(v) for k, v in s.value_counts(dropna=False).head(8).items()}
    R[f"ext_{name}"] = info
for d in ("heedb", "ed-bwh-ecg", "echonext/ECHONEXT", "lead"):
    fs = sorted(glob.glob(f"{HE}/data/{d}/**/*", recursive=True))
    tabs = [f for f in fs if f.lower().endswith((".csv", ".csv.gz", ".parquet", ".tsv", ".xlsx", ".json")) and os.path.getsize(f) < 5e9][:40]
    R[f"raw_{d}"] = {"n_files": len(fs), "tables": [f.replace(HE + "/data/", "") for f in tabs]}
    hdrs = {}
    for f in tabs[:25]:
        try:
            if f.endswith(".parquet"): hdrs[os.path.basename(f)] = list(pd.read_parquet(f).columns)[:40]
            elif f.endswith((".csv", ".csv.gz", ".tsv")): hdrs[os.path.basename(f)] = list(pd.read_csv(f, nrows=2, sep=None, engine="python").columns)[:40]
        except Exception as e: hdrs[os.path.basename(f)] = f"ERR {str(e)[:60]}"
    R[f"rawhdr_{d}"] = hdrs
json.dump(R, open(f"{OUT}/inventory.json", "w"), indent=1, default=str)
print("INVENTORY DONE")
