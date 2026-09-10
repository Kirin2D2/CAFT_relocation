"""Downstream-only blocks: does any rank block the cue once it is distributed?
base PCs, rho=1.0, seed 0. Kernel: %run -i experiments/run_layers3.py
"""
import sys, importlib
sys.path.insert(0, "/workspace/relocation/experiments")
import harness as H
import ablation, train_arm as _ta
importlib.reload(ablation); importlib.reload(_ta)
from ablation import load_directions, slice_dirs
from train_arm import train_arm
F = f"{H.ROOT}/data/directions/pca_base_gemma.pt"; PCA = load_directions(F)
JOBS = [("L5-25", list(range(5, 26)), 64), ("L5-25", list(range(5, 26)), 16), ("L5-25", list(range(5, 26)), 1),
        ("L1-25", list(range(1, 26)), 1)]
done = H.logged_run_ids()
for name, layers, k in JOBS:
    arm = f"pca_base_{name}_k{k}"; rid = f"arm_{arm}_rho1.0_s0"
    if rid in done: print(f"skip {rid}"); continue
    model = H.fresh_base(globals().get("model"))
    model, row, _ = train_arm(model, 1.0, 0, arm, dirs=slice_dirs(PCA, layers, k), dirs_file=F, complement=False)
    print(f"LAYERS3 {rid} n={len(layers)} k={k}: ID {row['id_accuracy']:.4f} OOD {row['ood_accuracy']:.4f} off {row['ood_accuracy_hooks_off']:.4f}")
model = H.fresh_base(globals().get("model"))
print("LAYERS3_STAGE_DONE")
