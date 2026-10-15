import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
import numpy as np, inspect, json, pandas as pd
H = f"{_HR}/cache/ntprobnp/ecg_data.npy"
x = np.memmap(H, dtype="float32", mode="r", shape=(18036, 12, 5000))
rng = np.random.default_rng(0); idx = np.sort(rng.choice(18036, 200, replace=False))
X = np.asarray(x[idx], dtype=np.float64)
def rel(a, b):
    return float(np.median(np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + 1e-9)))
I, II = X[:, 0], X[:, 1]
cand = {"III=II-I": (X[:, 2], II - I), "ch3=aVR=-(I+II)/2": (X[:, 3], -(I + II) / 2),
        "ch4=aVL=I-II/2": (X[:, 4], I - II / 2), "ch4=aVF=II-I/2": (X[:, 4], II - I / 2),
        "ch5=aVF=II-I/2": (X[:, 5], II - I / 2), "ch5=aVL=I-II/2": (X[:, 5], I - II / 2)}
R = {k: rel(a, b) for k, (a, b) in cand.items()}
R["amp_std_per_lead"] = [float(v) for v in X.std(axis=(0, 2))]
import ecg_audit.data.waveforms as W
src = inspect.getsource(W)
R["waveforms_file"] = W.__file__
keep = [l for l in src.splitlines() if any(k in l for k in ("CANON", "LEAD", "aVF", "aVL", "def load_canonical", "mimic", "emory", "heedb", "gain", "mV", "/ 1000", "/1000", "resample", "fs"))]
R["waveforms_lines"] = keep[:60]
# Emory record sample (for HARVI external input check)
from ecg_audit.data.locators import build_resolver
em = pd.read_parquet(f"{_PR}/manifests/manifest_emory.parquet").head(3)
res = build_resolver("emory", f"{_PR}/data", em)
s = W.load_canonical("emory", res(em.iloc[0])); R["emory_canonical_shape"] = list(s.shape); R["emory_canonical_std"] = [float(v) for v in s.std(axis=1)]
mm = pd.read_parquet(f"{_PR}/manifests/manifest_mimic_full.parquet").head(3)
rm = build_resolver("mimic", f"{_PR}/data", mm)
s2 = W.load_canonical("mimic", rm(mm.iloc[0])); R["mimic_canonical_shape"] = list(s2.shape); R["mimic_canonical_std"] = [float(v) for v in s2.std(axis=1)]
print(json.dumps(R, indent=1))
