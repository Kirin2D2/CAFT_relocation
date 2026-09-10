"""Relocation matrix orchestration. Kernel script: %run -i experiments/run_matrix.py
Set STAGE in the kernel first:  STAGE = "gate2" | "rho10" | "rho05" | "rho_mid"
Idempotent: skips any run whose final row is already in runs.jsonl.
Reassigns the kernel global `model` (fresh base per run).
"""
import sys, os
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H
from ablation import load_directions
from train_arm import train_arm

STAGE = globals().get("STAGE", "gate2")
DM_FILE = f"{H.ROOT}/data/directions/diffmeans_gemma.pt"
DM = load_directions(DM_FILE)
RND = {s: (f"{H.ROOT}/data/directions/random_gemma_s{s}.pt",
           load_directions(f"{H.ROOT}/data/directions/random_gemma_s{s}.pt")) for s in (0, 1)}
PROF_FILE = f"{H.ROOT}/data/directions/profession_gemma.pt"
PROF = load_directions(PROF_FILE)
BASE_ACTS = t.load(f"{H.ROOT}/data/acts/base_rho0.5_train.pt")["acts"]

PLAN = {
    "gate2":   [(1.0, 0, "ablated"), (1.0, 0, "ablated_prof")],
    "rho10":   [(1.0, 0, "random"), (1.0, 0, "plain"),
                (1.0, 1, "plain"), (1.0, 1, "ablated"), (1.0, 1, "random")],
    "rho05":   [(0.5, s, a) for s in (0, 1) for a in ("plain", "ablated", "random")],
    "rho_mid": [(r, s, a) for r in (0.8, 0.95) for s in (0, 1) for a in ("plain", "ablated", "random")],
}
done = H.logged_run_ids()
for rho, seed, arm in PLAN[STAGE]:
    rid = f"arm_{arm}_rho{rho}_s{seed}"
    if rid in done and f"{rid}_complement" in done:
        print(f"skip {rid} (logged)"); continue
    model = H.fresh_base(globals().get("model"))
    dirs, dfile = (None, None)
    if arm == "ablated":
        dirs, dfile = DM, DM_FILE
    elif arm == "ablated_prof":
        dirs, dfile = PROF, PROF_FILE
    elif arm == "random":
        dfile, dirs = RND[seed]
    model, row, crow = train_arm(model, rho, seed, arm, dirs=dirs, dirs_file=dfile,
                                 save=(STAGE == "gate2" and arm == "ablated"), base_acts=BASE_ACTS)
    if STAGE == "gate2" and arm == "ablated_prof":
        plain = [r for r in map(__import__("json").loads, open(H.RUNS))
                 if r["run_id"] == "gate1_gemma_rho1.0_s0"][0]
        print("\n================ GATE 2 (gemma-2-2b, rho=1.0, seed 0) ================")
        print(f"plain   (gate1_gemma_rho1.0_s0): ID {plain['id_accuracy']:.4f}  OOD {plain['ood_accuracy']:.4f}")
        import json as _j
        rows = {r["run_id"]: r for r in map(_j.loads, open(H.RUNS)) if not r.get("frontier")}
        for a in ("ablated", "ablated_prof"):
            r = rows[f"arm_{a}_rho1.0_s0"]; c = rows[f"arm_{a}_rho1.0_s0_complement"]; cp = rows[f"arm_{a}_rho1.0_s0_complement_pc"]
            print(f"{a:13} [{r['direction_source']}] hooks on : ID {r['id_accuracy']:.4f}  OOD {r['ood_accuracy']:.4f}")
            print(f"{'':13} hooks off: ID {r['id_accuracy_hooks_off']:.4f}  OOD {r['ood_accuracy_hooks_off']:.4f}")
            print(f"{'':13} +complement(diffmeans): ID {c['id_accuracy']:.4f}  OOD {c['ood_accuracy']:.4f}   +complement(pc): ID {cp['id_accuracy']:.4f}  OOD {cp['ood_accuracy']:.4f}")
        print("Rule: ID >= 0.95 and OOD meaningfully above plain's. Interpretation is the owner's.")
        print("GATE2_DONE")
model = H.fresh_base(globals().get("model"))
print(f"MATRIX_STAGE_DONE {STAGE}")
