"""Rank-k ablation subspaces for the intervention dose-response (pre-declared
2026-09-10 after Gate 2 failed at rank 1; owner delegated design).

For each source model (base gemma; plain-tuned rho=1.0 checkpoint), collect the
residual stream at ALL 26 layers on the rho=1.0 train set (626) at two positions:
the profession token (where the spurious cue enters) and the final token (where
the answer is read out). Per layer: pool both positions (1252 x 2304), centre,
SVD, rank the principal components by |corr(PC score, profession)|, keep the top
64 as an orthonormal U_l; mu_l = projection of the pooled mean. A run slices
(layers, k) from this. Matched random controls: QR of Gaussian [d, 64] per seed,
mu from the base acts. Requires kernel global `model` (clean base).
Run: %run -i experiments/directions_pca.py
"""
import sys, gc, json
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H
from transformers import AutoModelForCausalLM

assert "model" in globals()
KMAX = 64
ALL = list(range(model.config.num_hidden_layers))
TUNED = f"{H.ROOT}/data/checkpoints/gate1_gemma_rho1.0_s0/final"
ds = H.dataset(1.0); train = H.items(ds.train)
PID = {}
for p in ("doctor", "nurse"):
    e = H.tok.encode(" " + p, add_special_tokens=False); assert len(e) == 1, (p, e); PID[p] = e[0]
doc = t.tensor([c["profession"] == "doctor" for c in train]).float()


@t.no_grad()
def collect_two_pos(m, data, layers, batch=32):
    store = {l: {"final": [], "prof": []} for l in layers}
    cur = {}
    def mk(l):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[l]["final"].append(h[:, -1, :].float().cpu())
            store[l]["prof"].append(h[t.arange(h.shape[0], device=h.device), cur["pos"], :].float().cpu())
        return hook
    handles = [m.model.layers[l].register_forward_hook(mk(l)) for l in layers]
    try:
        for s in range(0, len(data), batch):
            chunk = data[s:s + batch]
            be = H.tok([c["formatted"] for c in chunk], padding=True, return_tensors="pt",
                       truncation=True, max_length=512).to(m.device)
            ids = be["input_ids"]
            pos = t.empty(ids.shape[0], dtype=t.long, device=ids.device)
            for j, c in enumerate(chunk):
                hits = (ids[j] == PID[c["profession"]]).nonzero().flatten()
                assert len(hits) >= 1, c["formatted"]
                pos[j] = hits[0]
            cur["pos"] = pos
            m(**be)
    finally:
        for h in handles: h.remove()
    return {l: {k: t.cat(v, 0) for k, v in d.items()} for l, d in store.items()}


def build(acts, source):
    y = t.cat([doc, doc]); yc = y - y.mean()
    dirs = {"layers": {}, "meta": {"source": f"pca_{source}", "model_id": H.MODEL_ID, "kmax": KMAX,
                                   "probe": "rho=1.0 train, profession token + final token pooled",
                                   "selection": "top |corr(PC score, profession)|", "stats": {}}}
    for l in ALL:
        X = t.cat([acts[l]["final"], acts[l]["prof"]], 0)
        mean = X.mean(0); Xc = X - mean
        _, S, Vh = t.linalg.svd(Xc, full_matrices=False)
        sc = Xc @ Vh.T                                          # [n, r]
        scc = sc - sc.mean(0)
        corr = (scc * yc[:, None]).sum(0) / (scc.norm(dim=0) * yc.norm() + 1e-8)
        order = corr.abs().argsort(descending=True)[:KMAX]
        U = Vh[order].T.contiguous()                            # [d, KMAX] orthonormal
        var_frac = (S[order] ** 2 / (S ** 2).sum())
        dirs["layers"][l] = {"U": U, "mu": (mean @ U)}
        dirs["meta"]["stats"][l] = {"abs_corr_top8": [round(float(corr[i].abs()), 3) for i in order[:8]],
                                    "abs_corr_k64": round(float(corr[order[-1]].abs()), 3),
                                    "var_frac_top8": [round(float(v), 4) for v in var_frac[:8]],
                                    "cum_var_k64": round(float(var_frac.sum()), 4)}
    return dirs


print("collecting base acts, 26 layers x 2 positions...")
acts_base = collect_two_pos(model, train, ALL)
dirs_base = build(acts_base, "base")
t.save(dirs_base, f"{H.ROOT}/data/directions/pca_base_gemma.pt")
for l in (0, 5, 10, 13, 16, 20, 25):
    print(f"base  L{l:2}: |corr| top8 {dirs_base['meta']['stats'][l]['abs_corr_top8']}  k64 {dirs_base['meta']['stats'][l]['abs_corr_k64']}  cumvar64 {dirs_base['meta']['stats'][l]['cum_var_k64']}")

for seed in (0, 1):
    g = t.Generator().manual_seed(2000 + seed)
    rd = {"layers": {}, "meta": {"source": "randk", "seed": seed, "kmax": KMAX, "model_id": H.MODEL_ID}}
    for l in ALL:
        d = acts_base[l]["final"].shape[1]
        Q, _ = t.linalg.qr(t.randn(d, KMAX, generator=g))
        X = t.cat([acts_base[l]["final"], acts_base[l]["prof"]], 0)
        rd["layers"][l] = {"U": Q.contiguous(), "mu": X.mean(0) @ Q}
    t.save(rd, f"{H.ROOT}/data/directions/randk_s{seed}_gemma.pt")
print("random rank-64 controls written (seeds 0,1)")
del acts_base; gc.collect()

print("loading tuned plain rho=1.0 checkpoint...")
tuned = AutoModelForCausalLM.from_pretrained(TUNED, device_map="cuda:0", dtype=t.bfloat16,
                                             attn_implementation="eager"); tuned.eval()
acts_tuned = collect_two_pos(tuned, train, ALL)
dirs_tuned = build(acts_tuned, "tuned")
t.save(dirs_tuned, f"{H.ROOT}/data/directions/pca_tuned_gemma.pt")
for l in (0, 5, 10, 13, 16, 20, 25):
    print(f"tuned L{l:2}: |corr| top8 {dirs_tuned['meta']['stats'][l]['abs_corr_top8']}  k64 {dirs_tuned['meta']['stats'][l]['abs_corr_k64']}  cumvar64 {dirs_tuned['meta']['stats'][l]['cum_var_k64']}")
# overlap of the top-1 base and tuned PCs per layer
print("cos(base PC1, tuned PC1) by layer:", [round(float(dirs_base["layers"][l]["U"][:, 0] @ dirs_tuned["layers"][l]["U"][:, 0]), 2) for l in ALL])
del tuned, acts_tuned; gc.collect(); t.cuda.empty_cache()
print("DIRECTIONS_PCA_DONE")
