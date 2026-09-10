"""Does the profession information survive the layer-0 block downstream?
Base model, rho=1.0 train set, final token AND profession token, layers 0/2/4/8/12/16/24:
d' of (doctor vs nurse) along the top profession-correlated direction of each layer's
acts, fit on those acts (LDA-free: diff-of-means projection), with vs without the
layer-0 rank-1 mean-ablation hook. Information, not use. Kernel: %run -i experiments/diag_layer0.py
"""
import sys, math, importlib, json
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H
import ablation; importlib.reload(ablation)
from ablation import load_directions, slice_dirs, MeanAblator
assert "model" in globals()
PCA = load_directions(f"{H.ROOT}/data/directions/pca_base_gemma.pt")
LAY = [0, 2, 4, 8, 12, 16, 24]
train = H.items(H.dataset(1.0).train)
doc = t.tensor([c["profession"] == "doctor" for c in train])
PID = {p: H.tok.encode(" " + p, add_special_tokens=False)[0] for p in ("doctor", "nurse")}

@t.no_grad()
def collect(m):
    store = {l: {"final": [], "prof": []} for l in LAY}; cur = {}
    def mk(l):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[l]["final"].append(h[:, -1, :].float().cpu())
            store[l]["prof"].append(h[t.arange(h.shape[0], device=h.device), cur["pos"], :].float().cpu())
        return hook
    hs = [m.model.layers[l].register_forward_hook(mk(l)) for l in LAY]
    try:
        for s in range(0, len(train), 32):
            ch = train[s:s+32]
            be = H.tok([c["formatted"] for c in ch], padding=True, return_tensors="pt").to(m.device)
            pos = t.tensor([(be["input_ids"][j] == PID[c["profession"]]).nonzero().flatten()[0] for j, c in enumerate(ch)], device=m.device)
            cur["pos"] = pos; m(**be)
    finally:
        for h in hs: h.remove()
    return {l: {k: t.cat(v) for k, v in d.items()} for l, d in store.items()}

def dprime(A):
    v = A[doc].mean(0) - A[~doc].mean(0); u = v / v.norm(); p = A @ u
    a, b = p[doc], p[~doc]
    return float((a.mean() - b.mean()) / math.sqrt((a.var() + b.var()) / 2)), float(v.norm())

res = {}
off = collect(model)
HOOK = globals().get("DIAG_HOOK", {"layers": [0], "k": 1})
abl = MeanAblator(slice_dirs(PCA, HOOK["layers"], HOOK["k"]), model).enable()   # registered AFTER collect's hooks are gone; collect re-registers after it -> fires after ablation
on = collect(model); abl.disable()
print(f"{'layer':>5} {'pos':>6} {'d_off':>7} {'|v|_off':>8} {'d_on':>7} {'|v|_on':>8}")
for l in LAY:
    for pos in ("prof", "final"):
        d0, n0 = dprime(off[l][pos]); d1, n1 = dprime(on[l][pos])
        res[f"L{l}_{pos}"] = {"dprime_off": round(d0, 2), "norm_off": round(n0, 2), "dprime_on": round(d1, 2), "norm_on": round(n1, 2)}
        print(f"{l:>5} {pos:>6} {d0:>7.2f} {n0:>8.2f} {d1:>7.2f} {n1:>8.2f}")
tag = f"L{HOOK['layers'][0]}-{HOOK['layers'][-1]}_k{HOOK['k']}"
print("DIAG_HOOK", tag)
json.dump(res, open(f"{H.ROOT}/data/diag_profinfo_{tag}.json", "w"), indent=1)
print("DIAG_LAYER0_DONE")
