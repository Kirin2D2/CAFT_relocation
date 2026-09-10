"""Relocation matrix at a chosen ablation dose. Kernel: set DOSE first, then
%run -i experiments/run_reloc.py
  DOSE = {"src": "pca_tuned"|"pca_base", "layers": "all"|"mid", "k": 16}
  RELOC_RHOS = [1.0, 0.5] (default) or [0.8, 0.95] etc.; RELOC_SEEDS = [0, 1]
Arms: plain, ablated (PCs at DOSE), random (randk_s{seed} at same layers/k).
Complement step ON (two rows per run). Idempotent on final+complement rows.
"""
import sys, importlib
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H
import ablation, train_arm as _ta
importlib.reload(ablation); importlib.reload(_ta)
from ablation import load_directions, slice_dirs
from train_arm import train_arm

DOSE = globals().get("DOSE", {"src": "pca_tuned", "layers": "all", "k": 16})
RHOS = globals().get("RELOC_RHOS", [1.0, 0.5])
SEEDS = globals().get("RELOC_SEEDS", [0, 1])
D = f"{H.ROOT}/data/directions"
LAYERS = list(range(26)) if DOSE["layers"] == "all" else [10, 13, 16]
k = DOSE["k"]
SRC_FILE = f"{D}/{DOSE['src']}_gemma.pt"
SRC = slice_dirs(load_directions(SRC_FILE), LAYERS, k)
RND = {s: (f"{D}/randk_s{s}_gemma.pt", slice_dirs(load_directions(f"{D}/randk_s{s}_gemma.pt"), LAYERS, k)) for s in SEEDS}
BASE_ACTS = t.load(f"{H.ROOT}/data/acts/base_rho0.5_train.pt")["acts"]
tag = f"{DOSE['src']}_L{len(LAYERS)}_k{k}"
done = H.logged_run_ids()
for rho in RHOS:
    for seed in SEEDS:
        for arm in ("plain", f"abl_{tag}", f"rnd_L{len(LAYERS)}_k{k}"):
            rid = f"arm_{arm}_rho{rho}_s{seed}"
            if rid in done and f"{rid}_complement_pc" in done:
                print(f"skip {rid}"); continue
            if arm == "plain":
                dirs, dfile = None, None
            elif arm.startswith("abl_"):
                dirs, dfile = SRC, SRC_FILE
            else:
                dfile, dirs = RND[seed]
            model = H.fresh_base(globals().get("model"))
            model, row, crow = train_arm(model, rho, seed, arm, dirs=dirs, dirs_file=dfile,
                                         base_acts=BASE_ACTS, complement=True)
            print(f"RELOC {rid}: ID {row['id_accuracy']:.4f} OOD {row['ood_accuracy']:.4f}"
                  + (f" off {row['ood_accuracy_hooks_off']:.4f}" if dirs else "")
                  + f" | +comp ID {crow['id_accuracy']:.4f} OOD {crow['ood_accuracy']:.4f}")
model = H.fresh_base(globals().get("model"))
print(f"RELOC_STAGE_DONE {tag} rhos={RHOS} seeds={SEEDS}")
