"""Layer-position sweep: which layers must carry the rank-1 block? rho=1.0, seed 0,
base PCs (fair source), k=1. Sets: thirds, halves, every-other, single layers.
Pre-declared 2026-09-10 01:50 UTC. No complement step. Kernel: %run -i experiments/run_layers.py
"""
import sys, importlib
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H
import ablation, train_arm as _ta
importlib.reload(ablation); importlib.reload(_ta)
from ablation import load_directions, slice_dirs
from train_arm import train_arm

F = f"{H.ROOT}/data/directions/pca_base_gemma.pt"
PCA = load_directions(F)
SETS = {"early0-8": list(range(0, 9)), "mid9-17": list(range(9, 18)), "late18-25": list(range(18, 26)),
        "half0-12": list(range(0, 13)), "half13-25": list(range(13, 26)), "even": list(range(0, 26, 2))}
for l in (0, 4, 8, 12, 16, 20, 24):
    SETS[f"only{l}"] = [l]
done = H.logged_run_ids()
for name, layers in SETS.items():
    arm = f"pca_base_{name}_k1"
    rid = f"arm_{arm}_rho1.0_s0"
    if rid in done:
        print(f"skip {rid}"); continue
    model = H.fresh_base(globals().get("model"))
    model, row, _ = train_arm(model, 1.0, 0, arm, dirs=slice_dirs(PCA, layers, 1), dirs_file=F, complement=False)
    print(f"LAYERS {rid} n={len(layers)}: ID {row['id_accuracy']:.4f} OOD {row['ood_accuracy']:.4f} off {row['ood_accuracy_hooks_off']:.4f}")
model = H.fresh_base(globals().get("model"))
print("LAYERS_STAGE_DONE")
