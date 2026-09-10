"""Follow-up to run_layers.py: second seed on the single-layer result; boundary layers.
Kernel: %run -i experiments/run_layers2.py
"""
import sys, importlib
sys.path.insert(0, "/workspace/relocation/experiments")
import harness as H
import ablation, train_arm as _ta
importlib.reload(ablation); importlib.reload(_ta)
from ablation import load_directions, slice_dirs
from train_arm import train_arm
F = f"{H.ROOT}/data/directions/pca_base_gemma.pt"; PCA = load_directions(F)
JOBS = [("only0", [0], 1), ("only4", [4], 1), ("early0-8", list(range(9)), 1),
        ("only1", [1], 0), ("only2", [2], 0), ("only6", [6], 0)]
done = H.logged_run_ids()
for name, layers, seed in JOBS:
    rid = f"arm_pca_base_{name}_k1_rho1.0_s{seed}"
    if rid in done: print(f"skip {rid}"); continue
    model = H.fresh_base(globals().get("model"))
    model, row, _ = train_arm(model, 1.0, seed, f"pca_base_{name}_k1", dirs=slice_dirs(PCA, layers, 1), dirs_file=F, complement=False)
    print(f"LAYERS2 {rid}: ID {row['id_accuracy']:.4f} OOD {row['ood_accuracy']:.4f} off {row['ood_accuracy_hooks_off']:.4f}")
model = H.fresh_base(globals().get("model"))
print("LAYERS2_STAGE_DONE")
