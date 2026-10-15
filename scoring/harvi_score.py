"""Usage: HARVI_DATA=/path/to/exports [HARVI_OUT=results] python scoring/harvi_score.py
Pre-specified HARVI scoring (harvi_preregistration.md sections 3-4). Patient-bootstrap, 2,000 draws, seed 0."""
import os, numpy as np, pandas as pd, json
DATA = os.environ.get("HARVI_DATA", "data"); OUTD = os.environ.get("HARVI_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
from sklearn.metrics import roc_auc_score
Z = np.load(f"{DATA}/score_export.npz"); EMm = pd.read_csv(f"{DATA}/score_emory_meta.csv")
D = pd.read_csv(f"{DATA}/harvi_oof_merged.csv")
REPS = 2000
y = D.ntprobnp.values; age = D.age.values
thr = np.where(age < 50, 450., np.where(age > 75, 1800., 900.))
LAB = np.stack([y < 300, y > thr, y > 3000], 1).astype(int)
HP = np.stack([D.harvi_p_ruleout, D.harvi_p_rulein, D.harvi_p_severe], 1)
MODELS = {"HARVI-ECG": HP, "FairECG-clin LoRA": Z["fairecgv2clin_lora_p"], "FairECG-clin frozen": Z["fairecgv2clin_frozen_p"],
          "FairECG-bal frozen": Z["fairecgv2bal_frozen_p"], "FairECG-rhy frozen": Z["fairecgv2rhy_frozen_p"]}

def groups(pid):
    _, inv = np.unique(pid, return_inverse=True); return [np.where(inv == i)[0] for i in range(inv.max() + 1)]
def boot_idx(G, rng): return np.concatenate([G[i] for i in rng.integers(0, len(G), len(G))])
def internal(mask, label):
    idx = np.where(mask)[0]; G = groups(D.subject_id.values[idx]); rng = np.random.default_rng(0)
    m2 = lambda P, ii: (roc_auc_score(LAB[ii, 0], P[ii, 0]) + roc_auc_score(LAB[ii, 1], P[ii, 1])) / 2
    pt = {m: [float(roc_auc_score(LAB[idx, h], P[idx, h])) for h in range(3)] + [float(m2(P, idx))] for m, P in MODELS.items()}
    bs = {m: [] for m in MODELS}
    for _ in range(REPS):
        b = idx[boot_idx(G, rng)]
        for m, P in MODELS.items(): bs[m].append([roc_auc_score(LAB[b, h], P[b, h]) for h in range(3)] + [m2(P, b)])
    rows = []
    for m in MODELS:
        B = np.array(bs[m]); dB = B - np.array(bs["HARVI-ECG"])
        for h, nm in enumerate(["rule-out", "rule-in (age-adj.)", "severe", "mean rule-out+rule-in"]):
            lo, hi = np.percentile(B[:, h], [2.5, 97.5]); dl, dh = np.percentile(dB[:, h], [2.5, 97.5])
            rows.append(dict(set=label, model=m, head=nm, auroc=pt[m][h], lo=lo, hi=hi,
                             d_vs_harvi=pt[m][h] - pt["HARVI-ECG"][h], d_lo=dl, d_hi=dh, n=len(idx), n_pat=len(G)))
    return rows

R = internal(np.ones(len(D), bool), "MIMIC OOF, all") + internal(D.in_fairecg_pretraining.values == 0, "MIMIC OOF, unexposed to FairECG pretraining")

# ---- external Emory, zero-shot, score = predicted NT-proBNP (mean of 5 fold models)
EXT = {"HARVI-ECG": Z["harvi_ext_mu"], "FairECG-clin LoRA": Z["fairecgv2clin_lora_ext_mu"]}
lab = EMm.label.values; ctl = EMm.ctl_for.fillna("").values; ea = EMm.age.values
ethr = np.where(ea < 50, 450., np.where(ea > 75, 1800., 900.))
E = []
for ph in ("hfref", "hfpef"):
    sel = np.where((lab == ph) | ((lab == "control") & (ctl == ph)))[0]; yy = (lab[sel] == ph).astype(int)
    G = groups(EMm.patient.values[sel]); rng = np.random.default_rng(0)
    pt = {m: roc_auc_score(yy, s[sel]) for m, s in EXT.items()}
    bs = {m: [] for m in EXT}
    for _ in range(REPS):
        b = boot_idx(G, rng)
        for m, s in EXT.items(): bs[m].append(roc_auc_score(yy[b], s[sel][b]))
    for m, s in EXT.items():
        B = np.array(bs[m]); dB = B - np.array(bs["HARVI-ECG"]); sc = s[sel]
        E.append(dict(endpoint=f"{ph.upper()} vs matched controls", model=m, auroc=pt[m], lo=np.percentile(B, 2.5), hi=np.percentile(B, 97.5),
                      d_vs_harvi=pt[m] - pt["HARVI-ECG"], d_lo=np.percentile(dB, 2.5), d_hi=np.percentile(dB, 97.5),
                      n_cases=int(yy.sum()), n_controls=int((1 - yy).sum()),
                      sens_at_300=float((sc[yy == 1] >= 300).mean()), spec_at_300=float((sc[yy == 0] < 300).mean()),
                      sens_at_ruleIn=float((sc[yy == 1] > ethr[sel][yy == 1]).mean()), spec_at_ruleIn=float((sc[yy == 0] <= ethr[sel][yy == 0]).mean()),
                      median_pred_case=float(np.median(sc[yy == 1])), median_pred_ctrl=float(np.median(sc[yy == 0]))))
sel = np.where(np.isin(lab, ["hfref", "hfpef"]))[0]; yy = (lab[sel] == "hfref").astype(int)
for m, s in EXT.items():
    E.append(dict(endpoint="HFrEF vs HFpEF (descriptive)", model=m, auroc=roc_auc_score(yy, s[sel]), n_cases=int(yy.sum()), n_controls=int((1 - yy).sum())))
pd.DataFrame(R).to_csv(f"{OUTD}/harvi_internal_results.csv", index=False); pd.DataFrame(E).to_csv(f"{OUTD}/harvi_external_results.csv", index=False)
print(pd.DataFrame(R).round(4).to_string()); print(pd.DataFrame(E).round(4).to_string())
