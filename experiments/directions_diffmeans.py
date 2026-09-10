"""DRAFT FOR USER REVIEW -- difference-of-means gender directions.

Spec (notes.md "Design decisions"): BASE model, rho=0.5 TRAIN set (626 items,
cues crossed by construction), residual stream at the final (scored) token,
grouped by gender of the CORRECT answer: u_l = normalize(mean_male - mean_female).
mu_l = mean over all 626 items of (h . u_l), for mean-ablation.
Computed once; reused identically at every rho.

Prints per layer: ||mean diff||, projection mean/std by gender, d' separation,
sign-accuracy of gender from the projection, and the PROFESSION contrast
projected onto u (doctor-mean - nurse-mean) . u -- should be ~0 at rho=0.5,
which is the check that u is gender and not profession.

Requires kernel globals `model` (clean base gemma). Run: %run -i experiments/directions_diffmeans.py
"""
import os, sys, json, math
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H

assert "model" in globals(), "need clean base gemma in kernel global `model`"
m0, _ = H.eval_split(model, H.dataset(1.0).test)
assert abs(m0["acc_2way"] - H.BASE_OOD_REF) < 0.02, f"base not clean: OOD {m0['acc_2way']:.4f}"
print(f"base OOD check {m0['acc_2way']:.4f} ok")

ds = H.dataset(0.5)
train = H.items(ds.train)
male = t.tensor([H.is_male(c) for c in train])
doc = t.tensor([c["profession"] == "doctor" for c in train])
print(f"rho=0.5 train n={len(train)}  male={int(male.sum())}  doctor={int(doc.sum())}  "
      f"male&doctor={int((male & doc).sum())} (crossed design => ~n/4)")

ACTS = f"{H.ROOT}/data/acts/base_rho0.5_train.pt"
if os.path.exists(ACTS):
    acts = t.load(ACTS)["acts"]
    print(f"loaded cached acts {ACTS}")
else:
    acts = H.collect_final_token(model, train, H.LAYERS)
    t.save({"acts": acts, "model_id": H.MODEL_ID, "layers": H.LAYERS, "rho": 0.5,
            "split": "train", "n": len(train),
            "items": [{"formatted": c["formatted"], "label": c["label"],
                       "profession": c["profession"], "category": c["category"]} for c in train]}, ACTS)
    print(f"cached acts -> {ACTS}")

dirs = {"layers": {}, "meta": {"source": "diff_of_means", "model_id": H.MODEL_ID,
                               "probe": "base, rho=0.5 train, final token, gender of correct answer",
                               "layers": H.LAYERS, "k": 1, "n": len(train)}}
stats = {}
us = {}
for l in H.LAYERS:
    A = acts[l]                                    # [n, d] fp32
    mm, mf = A[male].mean(0), A[~male].mean(0)
    diff = mm - mf
    u = diff / diff.norm()
    proj = A @ u
    pm, pf = proj[male], proj[~male]
    pooled = math.sqrt((pm.var() + pf.var()) / 2)
    dprime = float((pm.mean() - pf.mean()) / pooled)
    mid = float((pm.mean() + pf.mean()) / 2)
    sign_acc = float(((proj > mid) == male).float().mean())
    prof = (A[doc].mean(0) - A[~doc].mean(0))
    mu = float(proj.mean())
    us[l] = u
    dirs["layers"][l] = {"U": u[:, None].clone(), "mu": t.tensor([mu])}
    stats[l] = {"norm_diff": round(float(diff.norm()), 3),
                "norm_resid_median": round(float(A.norm(dim=1).median()), 1),
                "proj_male": [round(float(pm.mean()), 3), round(float(pm.std()), 3)],
                "proj_female": [round(float(pf.mean()), 3), round(float(pf.std()), 3)],
                "dprime": round(dprime, 3), "sign_acc": round(sign_acc, 4), "mu": round(mu, 3),
                "prof_contrast_dot_u": round(float(prof @ u), 3),
                "prof_contrast_norm": round(float(prof.norm()), 3)}
    print(f"L{l:2}: ||diff||={stats[l]['norm_diff']:>8}  |h|~{stats[l]['norm_resid_median']:>7}  "
          f"proj m {stats[l]['proj_male']} f {stats[l]['proj_female']}  d'={dprime:.2f}  "
          f"sign_acc={sign_acc:.3f}  mu={mu:.2f}  prof.u={stats[l]['prof_contrast_dot_u']} "
          f"(||prof||={stats[l]['prof_contrast_norm']})")
for i, a in enumerate(H.LAYERS):
    for b in H.LAYERS[i + 1:]:
        print(f"cos(u{a}, u{b}) = {float(us[a] @ us[b]):+.3f}")
dirs["meta"]["stats"] = stats

OUT = f"{H.ROOT}/data/directions/diffmeans_gemma.pt"
t.save(dirs, OUT)
print(f"wrote {OUT}  sha {H.sha_file(OUT)}")
print("DIRECTIONS_DIFFMEANS_DONE")
