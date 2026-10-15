import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
"""Why does the re-run OOF AUROC exceed the training log? Diagnostics, read-only.
(1) state_dict load report for each fold's SWA weights (strict=False hides missing keys)
(2) per fold: AUROC on its held-out fold vs on 3,600 of its own TRAINING rows (in-sample)
    -> if held-out ~= in-sample, the evaluation is leaking
(3) xlstm version in this environment
"""
import sys, os, json, numpy as np, torch, importlib.metadata as md
CS = sys.argv[1]; OUT = sys.argv[2]; DATA = sys.argv[3]
sys.path.insert(0, CS); os.chdir(CS)
from sklearn.metrics import roc_auc_score
from lightning_probnp import ProbNPLightningModule, ProbNPKFoldDataModule
from lightning_probnp.risk_config import threshold_array_for_head, apply_direction_numpy
import yaml
R = {"xlstm": md.version("xlstm"), "torch": torch.__version__, "lightning": md.version("pytorch-lightning")}
cfg = yaml.safe_load(open("configs/eval_oof_default.yaml")); tcfg = yaml.safe_load(open("configs/targets/ntprobnp.yaml"))
mk = dict(cfg["model"]); mk.update(tcfg.get("model", {}))
dk = dict(cfg["data"]); dk.update({"processed_data": DATA, "num_workers": 6})
dm = ProbNPKFoldDataModule(**{k: v for k, v in dk.items() if k in ProbNPKFoldDataModule.__init__.__code__.co_varnames})
dev = torch.device("cuda")
H = f"{_HR}/outputs/slurm"; JOBS = [68790, 68827, 68838, 68839, 68840]
torch.manual_seed(0); rng = np.random.default_rng(0)

def score(model, ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=6)
    P = []; Y = []; A = []
    with torch.no_grad():
        for b in loader:
            ecg, feat, raw_age, _, y_raw, _, _ = b
            ecg, feat = ecg.to(dev), feat.to(dev)
            st = (ecg.shape[-1] - model.crop_len) // 2
            out = model(ecg[..., st:st + model.crop_len], feat)       # deterministic centre crop only
            P.append(torch.sigmoid(out["risk_logits"]).cpu().numpy()); Y.append(y_raw.numpy().reshape(-1)); A.append(raw_age.numpy().reshape(-1))
    P = np.concatenate(P); Y = np.concatenate(Y); A = np.concatenate(A)
    res = {}
    for i, h in enumerate(model.risk_heads):
        yb = apply_direction_numpy(Y, threshold_array_for_head(h, A), str(h["direction"]))
        res[h["name"]] = float(roc_auc_score(yb, P[:, i]))
    res["avg"] = float(np.mean(list(res.values()))); return res

for k in range(5):
    dm.set_fold(k); dm.setup(stage="fit")
    model = ProbNPLightningModule(**{a: v for a, v in mk.items() if a in ProbNPLightningModule.__init__.__code__.co_varnames})
    sd = torch.load(f"{H}/{JOBS[k]}/checkpoints/fold_{k}_swa.pth", map_location="cpu")
    rep = model.model.load_state_dict(sd, strict=False)
    model.set_target_stats(dm.target_mean, dm.target_std); model.to(dev).eval()
    tr = np.asarray(dm.current_train_idx); va = np.asarray(dm.current_val_idx)
    sub = rng.choice(len(tr), size=len(va), replace=False)
    insample = torch.utils.data.Subset(dm.train_dataset, sub.tolist())
    R[f"fold{k}"] = {"missing_keys": len(rep.missing_keys), "unexpected_keys": len(rep.unexpected_keys),
                     "missing_sample": rep.missing_keys[:5], "unexpected_sample": rep.unexpected_keys[:5],
                     "heldout": score(model, dm.val_dataset), "insample_train": score(model, insample),
                     "train_ds_type": type(dm.train_dataset).__name__, "val_ds_type": type(dm.val_dataset).__name__}
    print(k, json.dumps(R[f"fold{k}"]), flush=True)
json.dump(R, open(f"{OUT}/diag.json", "w"), indent=1)
print("DIAG DONE")
