"""DRAFT FOR USER REVIEW -- post-fine-tuning directions in the complement of the
ablated subspace, for the test-time complement-ablation arm.

Probe set: rho=0.5 TRAIN (crossed design, so gender-of-correct-answer is
uncorrelated with profession for any tuned model). Activations are collected
with whatever hooks are currently enabled on the model (for ablated arms, the
training-time ablation stays on, so the acts are already the intervened ones).

method="diffmeans" (primary, direction_source "complement_diffmeans_tuned"):
    v = mean_male - mean_female on the TUNED model; v_c = v - U (U^T v);
    u_c = v_c / ||v_c||;  mu_c = mean(h . u_c) over the probe set.  Rank-matched (k=1).
method="pc_diff" (secondary, "complement_pc_diff"):
    D = acts_tuned - acts_base (same items); D_c = D - (D U) U^T; centre;
    u_c = top-1 right-singular vector of D_c, sign-aligned with v_c.
Reports ||v|| vs ||v_c|| (how much of the tuned gender contrast is in the
complement), d' along u_c on tuned and on base acts, cos(u_c, u_train).
Importable; no side effects.
"""
import math
import torch as t
import harness as H


def complement_directions(model, train_dirs, probe_items, method="diffmeans", base_acts=None):
    acts = H.collect_final_token(model, probe_items, H.LAYERS)
    male = t.tensor([H.is_male(c) for c in probe_items])
    dirs = {"layers": {}, "meta": {"source": f"complement_{method}_tuned" if method == "diffmeans"
                                             else "complement_pc_diff",
                                   "probe": "rho=0.5 train, final token", "layers": H.LAYERS, "k": 1}}
    stats = {}
    for l in H.LAYERS:
        A = acts[l]
        U = train_dirs["layers"][l]["U"].float()          # [d,k]
        u_train = U[:, 0] if U.shape[1] else None
        v = A[male].mean(0) - A[~male].mean(0)
        v_c = v - U @ (U.T @ v)
        if method == "diffmeans":
            u_c = v_c / v_c.norm()
        elif method == "pc_diff":
            assert base_acts is not None
            D = A - base_acts[l]
            D_c = D - (D @ U) @ U.T
            D_c = D_c - D_c.mean(0, keepdim=True)
            _, _, Vh = t.linalg.svd(D_c, full_matrices=False)
            u_c = Vh[0]
            if float(u_c @ v_c) < 0:
                u_c = -u_c
        else:
            raise ValueError(method)
        u_c = u_c - U @ (U.T @ u_c); u_c = u_c / u_c.norm()   # enforce complement exactly
        proj = A @ u_c
        pm, pf = proj[male], proj[~male]
        dprime = float((pm.mean() - pf.mean()) / math.sqrt((pm.var() + pf.var()) / 2))
        st = {"norm_v": round(float(v.norm()), 3), "norm_v_complement": round(float(v_c.norm()), 3),
              "frac_in_complement": round(float(v_c.norm() / v.norm()), 4) if float(v.norm()) > 0 else None,
              "dprime_tuned": round(dprime, 3), "cos_with_train_u": (round(float(u_c @ u_train), 5) if u_train is not None else None),
              "mu": round(float(proj.mean()), 3)}
        if base_acts is not None:
            bp = base_acts[l] @ u_c
            bm, bf = bp[male], bp[~male]
            st["dprime_base"] = round(float((bm.mean() - bf.mean()) / math.sqrt((bm.var() + bf.var()) / 2)), 3)
        stats[l] = st
        dirs["layers"][l] = {"U": u_c[:, None].clone(), "mu": t.tensor([float(proj.mean())])}
    dirs["meta"]["stats"] = stats
    return dirs, stats, acts
