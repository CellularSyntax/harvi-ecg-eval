import os as _os
_PR=_os.environ.get("PROJECT_ROOT","/path/to/project"); _FR=_os.environ.get("FAIRECG_ROOT","/path/to/fairecg"); _HR=_os.environ.get("HARVI_V2_ROOT","/path/to/harvi_v2")
"""FairECG backbone + HARVI multi-task heads on the HARVI MIMIC folds.

  --mode lora   : LoRA rank 8 on proj_up/proj_down (the paper campaign's recipe) + head
  --mode frozen : frozen backbone, embeddings cached once, same head and loss (ablation)
Heads (from harvi_v2/configs/targets/ntprobnp.yaml): rule-out <300, age-adjusted rule-in
(450/900/1800 for <50/50-75/>75 y), severe >3000; pos_weight 2/5/10; heteroscedastic Gaussian
NLL on train-standardised NT-proBNP. Age (train-standardised) and sex are given to the head.
Epoch selection: mean 3-head AUROC on an inner 10 % of TRAINING patients (never the outer fold).
Outputs: outer-fold predictions + predictions for every external ECG (results/harvi/external).
"""
import sys, os, json, math, argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, f"{_PR}"); sys.path.insert(0, f"{_PR}/scripts")
from fair_lora import build_bb, inject_lora, TARGETS_LORA

ap = argparse.ArgumentParser()
ap.add_argument("--alias", default="fairecgv2clin"); ap.add_argument("--ckpt", required=True)
ap.add_argument("--fold", type=int, required=True); ap.add_argument("--mode", choices=["lora", "frozen"], default="lora")
ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed + 100 * a.fold)
dev = torch.device("cuda")
HE = f"{_PR}"; HV = f"{HE}/results/harvi"
tag = f"{a.alias}_{a.mode}_fold{a.fold}"; os.makedirs(a.out, exist_ok=True)
log = lambda *s: print(f"[{tag} {time.strftime('%H:%M:%S')}]", *s, flush=True)

P = np.load(f"{HV}/harvi_pairs_folds.npz")
X = np.memmap(f"{_HR}/cache/ntprobnp/ecg_data.npy", dtype="float32", mode="r", shape=(18036, 12, 5000))
EX = np.load(f"{HV}/external/ext_ecg.npy", mmap_mode="r")
y = P["ntprobnp"].astype(np.float64); age = P["age"].astype(np.float64); fem = P["gender"].astype(np.float64)
fold = P["fold"]; sid = P["subject_id"]
import pandas as pd
EM = pd.read_parquet(f"{HV}/external/ext_meta.parquet")
eage = EM.age.values.astype(np.float64); efem = EM.female.values.astype(np.float64)

def thr_in(ag): return np.where(ag < 50, 450., np.where(ag > 75, 1800., 900.))
def labels(yv, ag): return np.stack([(yv < 300), (yv > thr_in(ag)), (yv > 3000)], 1).astype(np.float32)
L = labels(y, age)
tr_all = np.where(fold != a.fold)[0]; te = np.where(fold == a.fold)[0]
up = np.unique(sid[tr_all]); rng.shuffle(up); selp = set(up[: max(1, len(up) // 10)].tolist())
sel = np.array([i for i in tr_all if sid[i] in selp]); tr = np.array([i for i in tr_all if sid[i] not in selp])
assert not (set(sid[tr]) & set(sid[te])) and not (set(sid[sel]) & set(sid[te])) and not (set(sid[tr]) & set(sid[sel]))
NEXT = len(EM)
if a.smoke:
    tr, sel, te, NEXT = tr[:512], sel[:256], te[:256], 256; a.epochs = 1
ym, ys = y[tr].mean(), y[tr].std(); am, asd = age[tr].mean(), age[tr].std()
log(f"train {len(tr)} sel {len(sel)} test {len(te)} | patients disjoint OK | ext {len(EM)}")

bb = build_bb(a.alias, a.ckpt, None, dev)
for p in bb._model.parameters(): p.requires_grad = False
nl = inject_lora(bb._model, TARGETS_LORA[a.alias], a.rank) if a.mode == "lora" else 0
bb._model.to(dev)
D = bb.embed_dim
head = nn.Sequential(nn.Linear(D + 2, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 5)).to(dev)
params = [p for p in bb._model.parameters() if p.requires_grad] + list(head.parameters())
log(f"LoRA layers {nl}; trainable {sum(p.numel() for p in params)/1e3:.1f}K; D={D}")
POSW = torch.tensor([2., 5., 10.], device=dev)

def tab(ag, fe): return torch.tensor(np.stack([(ag - am) / asd, fe], 1), dtype=torch.float32, device=dev)
def emb(src, idx):
    xb = torch.from_numpy(np.nan_to_num(np.asarray(src[idx], dtype=np.float32))).to(dev)
    return torch.nan_to_num(bb.embed(xb, fs=500))
def loss_fn(o, yb, lb):
    mu, lv = o[:, 0], o[:, 1].clamp(-7, 7); t = (yb - ym) / ys
    nll = 0.5 * (lv + (t - mu) ** 2 / torch.exp(lv)).mean()
    bce = F.binary_cross_entropy_with_logits(o[:, 2:], lb, pos_weight=POSW)
    return nll + 3 * bce   # mean over 3 heads x3 = sum of per-head BCE (HARVI loss_weight 1 each)

Zc = {}
if a.mode == "frozen":
    @torch.no_grad()
    def cache(src, idx, bs=64):
        out = np.zeros((len(idx), D), np.float32); bb._model.eval()
        for k in range(0, len(idx), bs): out[k:k + bs] = emb(src, idx[k:k + bs]).float().cpu().numpy()
        return out
    zf = f"{a.out}/zcache_{a.alias}.npz"
    if os.path.exists(zf):
        z = np.load(zf); Zc["harvi"], Zc["ext"] = z["harvi"], z["ext"]; log("embeddings loaded from cache")
    else:
        Zc["harvi"] = cache(X, np.arange(18036)); Zc["ext"] = cache(EX, np.arange(len(EM)))
        np.savez(zf + ".tmp.npz", harvi=Zc["harvi"], ext=Zc["ext"]); os.replace(zf + ".tmp.npz", zf); log("embeddings cached to disk")

def forward(src, idx, ag, fe, train):
    if a.mode == "frozen":
        z = torch.from_numpy(Zc["harvi" if src is X else "ext"][idx]).to(dev)
    else:
        z = emb(src, idx)
    return head(torch.cat([z, tab(ag, fe)], 1))

@torch.no_grad()
def predict(src, idx, ag, fe, bs=64):
    bb._model.eval(); head.eval(); O = []
    for k in range(0, len(idx), bs):
        O.append(forward(src, idx[k:k + bs], ag[k:k + bs], fe[k:k + bs], False).float().cpu().numpy())
    O = np.concatenate(O); return O

def score(O, lb):
    pr = 1 / (1 + np.exp(-O[:, 2:])); return [float(roc_auc_score(lb[:, h], pr[:, h])) for h in range(3)]

opt = torch.optim.AdamW(params, lr=a.lr if a.mode == "lora" else 1e-3, weight_decay=1e-4)
spe = math.ceil(len(tr) / a.batch); tot = spe * a.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / spe if s < spe else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - spe) / max(1, tot - spe)))))
best = (-1, None, -1); hist = []
for ep in range(a.epochs):
    bb._model.train() if a.mode == "lora" else bb._model.eval(); head.train()
    order = rng.permutation(tr); tl = 0; nb = 0
    for k in range(0, len(order), a.batch):
        bi = np.sort(order[k:k + a.batch])
        o = forward(X, bi, age[bi], fem[bi], True)
        loss = loss_fn(o, torch.tensor(y[bi], dtype=torch.float32, device=dev), torch.from_numpy(L[bi]).to(dev))
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        tl += float(loss); nb += 1
    s = score(predict(X, sel, age[sel], fem[sel]), L[sel]); m = float(np.mean(s)); hist.append((ep, tl / max(nb, 1), s))
    log(f"epoch {ep} loss {tl/max(nb,1):.4f} sel AUROC {[round(v,4) for v in s]} mean {m:.4f}")
    if m > best[0]:
        best = (m, {"head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                    "lora": {n: p.detach().clone() for n, p in bb._model.named_parameters() if p.requires_grad}}, ep)
head.load_state_dict(best[1]["head"])
sd = dict(bb._model.named_parameters())
for n, v in best[1]["lora"].items(): sd[n].data.copy_(v)
Ot = predict(X, te, age[te], fem[te]); st = score(Ot, L[te])
Oe = predict(EX, np.arange(NEXT), eage[:NEXT], efem[:NEXT])
np.savez(f"{a.out}/pred_{tag}.npz", test_idx=te, test_mu_pg=Ot[:, 0] * ys + ym, test_logvar=Ot[:, 1], test_p=1 / (1 + np.exp(-Ot[:, 2:])),
         ext_mu_pg=Oe[:, 0] * ys + ym, ext_p=1 / (1 + np.exp(-Oe[:, 2:])), sel_idx=sel)
torch.save(best[1], f"{a.out}/state_{tag}.pt")
json.dump({"tag": tag, "best_epoch": best[2], "best_sel_mean": best[0], "test_auroc_ruleout_rulein_severe": st,
           "test_mean3": float(np.mean(st)), "n_train": len(tr), "n_sel": len(sel), "n_test": len(te), "lora_layers": nl,
           "history": hist}, open(f"{a.out}/res_{tag}.json", "w"), indent=1)
log(f"DONE best epoch {best[2]} test AUROC {[round(v,4) for v in st]}")
