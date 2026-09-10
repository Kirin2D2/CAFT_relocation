"""Intervention dose-response at fixed rho. Kernel: %run -i experiments/run_dose.py
Conditions: source in {pca_base, pca_tuned, randk(seed 0)} x k in {64,16,4,1} at all 26
layers, rho=1.0 then rho=0.5 (grammar-learning control), seed 0; then pca_tuned at the
3 mid layers for k in {64,16,4}, rho=1.0. No complement step (speed). Idempotent.
"""
import sys
sys.path.insert(0, "/workspace/relocation/experiments")
import importlib, torch as t
import harness as H
import ablation, train_arm as _ta
importlib.reload(ablation); importlib.reload(_ta)   # kernel caches modules across %run
from ablation import load_directions, slice_dirs
from train_arm import train_arm

D = f"{H.ROOT}/data/directions"
SRC = {"pca_base": load_directions(f"{D}/pca_base_gemma.pt"),
       "pca_tuned": load_directions(f"{D}/pca_tuned_gemma.pt"),
       "randk": load_directions(f"{D}/randk_s0_gemma.pt")}
FILE = {"pca_base": f"{D}/pca_base_gemma.pt", "pca_tuned": f"{D}/pca_tuned_gemma.pt", "randk": f"{D}/randk_s0_gemma.pt"}
ALL = list(range(26)); MID = [10, 13, 16]
conds = []
for rho in (1.0, 0.5):
    for k in (64, 16, 4, 1):
        for src in ("pca_tuned", "pca_base", "randk"):
            conds.append((rho, src, ALL, k))
for k in (64, 16, 4):
    conds.append((1.0, "pca_tuned", MID, k))
done = H.logged_run_ids()
for rho, src, layers, k in conds:
    arm = f"{src}_L{len(layers)}_k{k}"
    rid = f"arm_{arm}_rho{rho}_s0"
    if rid in done:
        print(f"skip {rid}"); continue
    dirs = slice_dirs(SRC[src], layers, k)
    model = H.fresh_base(globals().get("model"))
    model, row, _ = train_arm(model, rho, 0, arm, dirs=dirs, dirs_file=FILE[src], complement=False)
    print(f"DOSE {rid}: ID {row['id_accuracy']:.4f} OOD {row['ood_accuracy']:.4f} off {row['ood_accuracy_hooks_off']:.4f}")
model = H.fresh_base(globals().get("model"))
print("DOSE_STAGE_DONE")
