import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
import numpy as np, pandas as pd, sys
OUT = f"{_PR}/results/harvi"
p = np.load(f"{OUT}/harvi_pairs_folds.npz"); inpre = np.load(f"{OUT}/harvi_pair_in_pretraining.npy")
o = pd.read_csv(f"{OUT}/oof_harvi/oof_predictions.csv")
assert len(o) == len(p["ntprobnp"]) and (o.sample_index.values == np.arange(len(o))).all()
assert np.allclose(o.target.values, p["ntprobnp"]), "label mismatch"
df = pd.DataFrame(dict(idx=np.arange(len(o)), subject_id=p["subject_id"], fold=p["fold"], age=p["age"], female=p["gender"],
                       ntprobnp=p["ntprobnp"], in_fairecg_pretraining=inpre.astype(int),
                       harvi_pred_ntprobnp=o.pred_reg, harvi_p_ruleout=o.pred_head_low_risk,
                       harvi_p_rulein=o.pred_head_high_risk, harvi_p_severe=o.pred_head_severe_risk))
df.to_csv(f"{OUT}/harvi_oof_merged.csv", index=False)
print("MERGED", len(df), df.subject_id.nunique())
