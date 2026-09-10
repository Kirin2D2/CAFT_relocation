"""Secondary, pre-declared variant: profession-contrast direction on the base model.
u_l = normalize(mean_doctor - mean_nurse) on the cached rho=0.5 train base acts;
mu_l = mean(h . u_l). Blocks the stereotype's INPUT axis rather than the
gender-of-answer axis. Declared secondary before any Gate 2 result was seen
(2026-09-10); headline remains diff-of-means. Standalone, from the acts cache.
"""
import sys, math
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H

cache = t.load(f"{H.ROOT}/data/acts/base_rho0.5_train.pt")
acts, items = cache["acts"], cache["items"]
doc = t.tensor([c["profession"] == "doctor" for c in items])
male = t.tensor([c["label"] in ("he", "him") for c in items])
dm = t.load(f"{H.ROOT}/data/directions/diffmeans_gemma.pt", weights_only=False)
dirs = {"layers": {}, "meta": {"source": "profession_contrast", "model_id": H.MODEL_ID,
                               "probe": "base, rho=0.5 train, final token, doctor minus nurse",
                               "layers": H.LAYERS, "k": 1, "n": len(items), "stats": {}}}
for l in H.LAYERS:
    A = acts[l]
    v = A[doc].mean(0) - A[~doc].mean(0)
    u = v / v.norm()
    p = A @ u
    pd, pn = p[doc], p[~doc]
    dprime = float((pd.mean() - pn.mean()) / math.sqrt((pd.var() + pn.var()) / 2))
    mid = float((pd.mean() + pn.mean()) / 2)
    sign_acc = float(((p > mid) == doc).float().mean())
    gender_dot = float((A[male].mean(0) - A[~male].mean(0)) @ u)
    cos = float(u @ dm["layers"][l]["U"][:, 0])
    st = {"norm_v": round(float(v.norm()), 3), "dprime": round(dprime, 3), "sign_acc": round(sign_acc, 4),
          "mu": round(float(p.mean()), 3), "gender_contrast_dot_u": round(gender_dot, 3),
          "cos_with_diffmeans": round(cos, 4)}
    dirs["layers"][l] = {"U": u[:, None].clone(), "mu": t.tensor([float(p.mean())])}
    dirs["meta"]["stats"][l] = st
    print(f"L{l:2}: {st}")
out = f"{H.ROOT}/data/directions/profession_gemma.pt"
t.save(dirs, out)
print(f"wrote {out}  sha {H.sha_file(out)}")
print("DIRECTIONS_PROFESSION_DONE")
