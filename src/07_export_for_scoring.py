import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
"""Compact export for scoring: internal OOF arrays for FairECG LoRA (+ frozen ablation) and Emory-only external scores."""
import numpy as np, pandas as pd, json
HV = f"{_PR}/results/harvi"; FO = f"{HV}/fairecg"
EM = pd.read_parquet(f"{HV}/external/ext_meta.parquet"); em = np.where(EM.set.values == "emory")[0]
out = {"emory_idx": em}
def oof(tag):
    p = np.full((18036, 3), np.nan); mu = np.full(18036, np.nan); ext_mu = []; ext_p = []
    for k in range(5):
        z = np.load(f"{FO}/pred_{tag}_fold{k}.npz")
        p[z["test_idx"]] = z["test_p"]; mu[z["test_idx"]] = z["test_mu_pg"]; ext_mu.append(z["ext_mu_pg"]); ext_p.append(z["ext_p"])
    assert np.isfinite(p).all(), tag
    return p, mu, np.mean(ext_mu, 0)[em], np.mean(ext_p, 0)[em]
for tag in ("fairecgv2clin_lora", "fairecgv2clin_frozen", "fairecgv2bal_frozen", "fairecgv2rhy_frozen"):
    p, mu, emu, ep = oof(tag)
    out[f"{tag}_p"] = p; out[f"{tag}_mu"] = mu; out[f"{tag}_ext_mu"] = emu; out[f"{tag}_ext_p"] = ep
h = np.load(f"{HV}/external/pred_harvi_ext.npz")
out["harvi_ext_mu"] = h["reg"][em]; out["harvi_ext_p"] = h["risk"][em]
np.savez_compressed(f"{HV}/score_export.npz", **out)
EM.iloc[em][["patient", "label", "ctl_for", "age", "female"]].to_csv(f"{HV}/score_emory_meta.csv", index=False)
print("EXPORT OK", len(em))
