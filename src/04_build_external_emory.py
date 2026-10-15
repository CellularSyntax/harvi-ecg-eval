import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
"""Build the pre-specified external test sets and cache their ECGs once (canonical [12,5000] @500 Hz, mV).
Emory (HEEDB I0006): HFrEF / HFpEF / mixed case ECGs + 4:1 controls matched on sex and 5-year age band,
                     drawn without replacement (seed 0), separately for HFrEF and HFpEF.
                     DATA CORRECTION (declared before scoring): stored aVL/aVF are -III/2 and +III/2;
                     rebuilt as aVL = I - II/2, aVF = II - I/2 (I, II, III, aVR verified correct).
EchoNext:            one ECG per patient (seed 0) with echo LVEF <= 40 % (case) or >= 50 % (control).
"""
import numpy as np, pandas as pd, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, f"{_PR}")
import ecg_audit.data.waveforms as W
from ecg_audit.data.locators import build_resolver
HE = f"{_PR}"; OUT = f"{HE}/results/harvi/external"; E = f"{HE}/data/heedb/I0006"
os.makedirs(OUT, exist_ok=True); R = {}
rng = np.random.default_rng(0)

# ---------------- Emory matched case-control ----------------
cs = pd.read_parquet(f"{HE}/results/harvi/emory_cases_draft.parquet"); ct = pd.read_parquet(f"{HE}/results/harvi/emory_controls_draft.parquet")
cs = cs.dropna(subset=["AgeAtAcquisition"]); ct = ct.dropna(subset=["AgeAtAcquisition"])
for d in (cs, ct):
    d["female"] = (d.Sex.astype(str).str.upper() == "FEMALE").astype(int); d["band"] = (d.AgeAtAcquisition // 5).astype(int)
cs = cs[cs.AgeAtAcquisition.between(18, 100)]; ct = ct[ct.AgeAtAcquisition.between(18, 100)]
pool = ct.sample(frac=1.0, random_state=0).copy(); pool["used"] = ""
rows = []
for ph in ("hfref", "hfpef"):
    c = cs[cs.phenotype == ph]; short = 0
    for (f, b), g in c.groupby(["female", "band"]):
        avail = pool.index[(pool.female == f) & (pool.band == b) & (pool.used == "")]
        k = min(4 * len(g), len(avail)); short += 4 * len(g) - k
        pool.loc[avail[:k], "used"] = ph
    R[f"emory_{ph}_cases"] = int(len(c)); R[f"emory_{ph}_controls"] = int((pool.used == ph).sum()); R[f"emory_{ph}_control_shortfall"] = int(short)
emory = pd.concat([cs.assign(set="emory", label=cs.phenotype, ctl_for=""),
                   pool[pool.used != ""].assign(set="emory", label="control", ctl_for=pool.used[pool.used != ""])], ignore_index=True)
emory["path"] = E + "/" + emory.FileName.astype(str); emory["patient"] = emory.BDSPPatientID.astype(str)
emory["age"] = emory.AgeAtAcquisition.astype(float); emory["lvef"] = np.nan
emory = emory[["set", "patient", "path", "label", "ctl_for", "age", "female", "lvef"]]

# ---------------- EchoNext ----------------
man = pd.read_parquet(f"{HE}/manifests/manifest_echonext.parquet")
meta = pd.read_csv(f"{HE}/data/echonext/ECHONEXT/echonext_metadata_100k.csv", low_memory=False)
assert len(man) == len(meta) == 100000
if "split" in man.columns: assert (man.split.astype(str).values == meta.split.astype(str).values).all(), "manifest/metadata order differs"
man = man.reset_index(drop=True); res_en = build_resolver("echonext", f"{HE}/data", man)
man["lvef"] = meta.lvef_value.values; man["patient"] = meta.patient_key.astype(str).values
man["age"] = pd.to_numeric(meta.age_at_ecg, errors="coerce").values; man["female"] = (meta.sex.astype(str).str.upper().str.startswith("F")).astype(int).values
el = man[(man.lvef <= 40) | (man.lvef >= 50)].dropna(subset=["age"])
el = el.sample(frac=1.0, random_state=0).groupby("patient").head(1)
R["echonext_cases_lvef_le40"] = int((el.lvef <= 40).sum()); R["echonext_controls_lvef_ge50"] = int((el.lvef >= 50).sum())
el_loc = [res_en(man.loc[i]) for i in el.index]
echo = pd.DataFrame(dict(set="echonext", patient=el.patient.values, path=[json.dumps(l) for l in el_loc],
                         label=np.where(el.lvef.values <= 40, "lvef_le40", "lvef_ge50"), ctl_for="",
                         age=el.age.values.astype(float), female=el.female.values, lvef=el.lvef.values))

ext = pd.concat([emory, echo], ignore_index=True)
N = len(ext); R["n_total"] = N
mm = np.lib.format.open_memmap(f"{OUT}/ext_ecg.npy", mode="w+", dtype=np.float32, shape=(N, 12, 5000))

def load(i):
    r = ext.iloc[i]
    try:
        if r.set == "emory":
            s = W.load_canonical("emory", {"record_path": r.path})
            I, II = s[0], s[1]; s = s.copy(); s[4] = I - II / 2.0; s[5] = II - I / 2.0
        else:
            s = W.load_canonical("echonext", json.loads(r.path))
        if s.shape[1] != 5000:
            s = s[:, :5000] if s.shape[1] > 5000 else np.tile(s, (1, int(np.ceil(5000 / s.shape[1]))))[:, :5000]
        return i, np.nan_to_num(s.astype(np.float32)), True
    except Exception as e:
        return i, None, str(e)[:80]

ok = np.zeros(N, bool); errs = {}
with Pool(14) as p:
    for k, (i, s, st) in enumerate(p.imap_unordered(load, range(N), chunksize=64)):
        if s is not None: mm[i] = s; ok[i] = True
        else: errs[st] = errs.get(st, 0) + 1
        if k % 10000 == 0: print("loaded", k, flush=True)
mm.flush(); ext["ok"] = ok
ext.to_parquet(f"{OUT}/ext_meta.parquet", index=False)
R["load_ok"] = int(ok.sum()); R["load_errors"] = errs
# integrity: Einthoven on 200 cached Emory rows + amplitude per set
ix = np.where((ext.set == "emory") & ok)[0][:200]; X = np.asarray(mm[ix], dtype=np.float64)
nr = lambda a, b: float(np.median(np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + 1e-9)))
R["emory_einthoven_after_fix"] = {"III": nr(X[:, 2], X[:, 1] - X[:, 0]), "aVL": nr(X[:, 4], X[:, 0] - X[:, 1] / 2), "aVF": nr(X[:, 5], X[:, 1] - X[:, 0] / 2)}
for st in ("emory", "echonext"):
    ix = np.where((ext.set == st) & ok)[0][:300]; R[f"amp_std_{st}"] = [round(float(v), 4) for v in np.asarray(mm[ix]).std(axis=(0, 2))]
json.dump(R, open(f"{OUT}/build.json", "w"), indent=1); print(json.dumps(R, indent=1))
