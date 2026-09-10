"""Matched-rank random-direction control. One random unit vector per layer per
seed (k=1, matching diff-of-means); mu from the same cached base rho=0.5 train
activations, so the intervention differs from the real arm ONLY in direction.
Standalone once data/acts/base_rho0.5_train.pt exists. Run: %run -i experiments/directions_random.py
"""
import sys
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H

SEEDS = [0, 1]
acts = t.load(f"{H.ROOT}/data/acts/base_rho0.5_train.pt")["acts"]
dm = t.load(f"{H.ROOT}/data/directions/diffmeans_gemma.pt", weights_only=False)
for seed in SEEDS:
    g = t.Generator().manual_seed(1000 + seed)
    dirs = {"layers": {}, "meta": {"source": "random_matched_rank", "seed": seed,
                                   "model_id": H.MODEL_ID, "layers": H.LAYERS, "k": 1}}
    for l in H.LAYERS:
        d = acts[l].shape[1]
        u = t.randn(d, generator=g); u = u / u.norm()
        mu = float((acts[l] @ u).mean())
        cos = float(u @ dm["layers"][l]["U"][:, 0])
        proj = acts[l] @ u
        dirs["layers"][l] = {"U": u[:, None], "mu": t.tensor([mu])}
        print(f"seed {seed} L{l:2}: mu={mu:+.3f} proj std={float(proj.std()):.3f} cos(random, diffmeans)={cos:+.4f}")
    out = f"{H.ROOT}/data/directions/random_gemma_s{seed}.pt"
    t.save(dirs, out)
    print(f"wrote {out}  sha {H.sha_file(out)}")
print("DIRECTIONS_RANDOM_DONE")
