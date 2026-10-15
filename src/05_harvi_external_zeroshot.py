import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
"""Score the external ECG cache with the five existing HARVI-ECG fold models (SWA weights), zero-shot.
Per fold k the inputs are prepared exactly as for that fold's validation data in harvi_v2:
ECG (x - global_mean)/(global_std+1e-6) with the DataModule's global stats, age standardised with a
StandardScaler fit on fold-k training ages, sex 1=female; inference = harvi_v2 _infer_fold (10 TTA crops,
calibrate_mu). Final score = mean over the 5 fold models.
"""
import sys, os, json, numpy as np, torch, yaml, pytorch_lightning as pl
CS = sys.argv[1]; HV = sys.argv[2]; DATA = sys.argv[3]
sys.path.insert(0, CS); os.chdir(CS)
from lightning_probnp import ProbNPLightningModule, ProbNPKFoldDataModule
from evaluate_oof_lightning import _infer_fold
from sklearn.preprocessing import StandardScaler
import pandas as pd
pl.seed_everything(42, workers=True)
cfg = yaml.safe_load(open("configs/eval_oof_default.yaml")); tcfg = yaml.safe_load(open("configs/targets/ntprobnp.yaml"))
mk = dict(cfg["model"]); mk.update(tcfg.get("model", {}))
dk = dict(cfg["data"]); dk.update({"processed_data": DATA, "num_workers": 2})
dm = ProbNPKFoldDataModule(**{k: v for k, v in dk.items() if k in ProbNPKFoldDataModule.__init__.__code__.co_varnames})
dev = torch.device("cuda")
EX = np.load(f"{HV}/external/ext_ecg.npy", mmap_mode="r"); EM = pd.read_parquet(f"{HV}/external/ext_meta.parquet")
eage = EM.age.values.astype(np.float32); efem = EM.female.values.astype(np.float32)
CK = f"{HV}/ckpts"

class ExtDS(torch.utils.data.Dataset):
    def __init__(self, gm, gs, tab): self.gm, self.gs, self.tab = gm, gs, tab
    def __len__(self): return len(EM)
    def __getitem__(self, i):
        s = torch.from_numpy(np.asarray(EX[i], dtype=np.float32).copy()); s = (s - self.gm) / (self.gs + 1e-6)
        return (s, torch.tensor(self.tab[i], dtype=torch.float32), torch.tensor(eage[i]), torch.tensor(0.), torch.tensor(0.),
                torch.zeros(6, dtype=torch.int8), torch.tensor(0))

REG = []; RISK = []; meta = {}
for k in range(5):
    dm.set_fold(k); dm.setup(stage="fit")
    ages = dm.metadata["age"].reshape(-1, 1)
    sc = StandardScaler().fit(ages[dm.current_train_idx])
    tab = np.stack([sc.transform(eage.reshape(-1, 1)).reshape(-1), efem], 1).astype(np.float32)
    model = ProbNPLightningModule(**{a: v for a, v in mk.items() if a in ProbNPLightningModule.__init__.__code__.co_varnames})
    rep = model.model.load_state_dict(torch.load(f"{CK}/fold_{k}_swa.pth", map_location="cpu"), strict=False)
    assert not rep.missing_keys and not rep.unexpected_keys, rep
    model.set_target_stats(dm.target_mean, dm.target_std); model.to(dev).eval()
    dl = torch.utils.data.DataLoader(ExtDS(dm.global_mean, dm.global_std, tab), batch_size=64, shuffle=False, num_workers=0)
    out = _infer_fold(model=model, dataloader=dl, tta_crops=10, conf_thresh=0.8)
    REG.append(out["pred_reg"]); RISK.append(out["pred_risk"])
    meta[k] = {"global_mean": float(dm.global_mean), "global_std": float(dm.global_std), "target_mean": float(dm.target_mean)}
    print("fold", k, "done", meta[k], flush=True)
np.savez(f"{HV}/external/pred_harvi_ext.npz", reg_folds=np.stack(REG), risk_folds=np.stack(RISK),
         reg=np.mean(REG, 0), risk=np.mean(RISK, 0))
json.dump(meta, open(f"{HV}/external/pred_harvi_ext_meta.json", "w"), indent=1)
print("HARVI EXTERNAL DONE", flush=True)