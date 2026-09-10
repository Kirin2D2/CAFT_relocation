"""Hook correctness on the base model, one batch. Run: %run -i experiments/smoke_ablation.py
1. With a MeanAblator on random unit directions (mu=0.3), the collected final-token
   activations satisfy h.u == mu (relative to ||h||, bf16 tolerance) at every hooked layer.
2. Disabling the hooks restores the unhooked logits exactly.
3. Base OOD under random-direction ablation is reported (should be ~unchanged).
"""
import sys
sys.path.insert(0, "/workspace/relocation/experiments")
import torch as t
import harness as H
from ablation import MeanAblator

assert "model" in globals()
data = H.items(H.dataset(1.0).test)[:32]
d = model.config.hidden_size
g = t.Generator().manual_seed(123)
dirs = {"layers": {}, "meta": {"source": "smoke"}}
for l in H.LAYERS:
    u = t.randn(d, generator=g); u = u / u.norm()
    dirs["layers"][l] = {"U": u[:, None], "mu": t.tensor([0.3])}

be = H.tok([c["formatted"] for c in data], padding=True, return_tensors="pt").to(model.device)
with t.no_grad():
    ref = model(**be).logits[:, -1, :].float().clone()
acts0 = H.collect_final_token(model, data, H.LAYERS)
abl = MeanAblator(dirs, model).enable()
acts1 = H.collect_final_token(model, data, H.LAYERS)
with t.no_grad():
    hooked = model(**be).logits[:, -1, :].float().clone()
abl.disable()
with t.no_grad():
    back = model(**be).logits[:, -1, :].float().clone()

ok = True
for l in H.LAYERS:
    u = dirs["layers"][l]["U"][:, 0]
    p0 = acts0[l] @ u; p1 = acts1[l] @ u
    hn = acts1[l].norm(dim=1)
    rel = ((p1 - 0.3).abs() / hn).max()
    print(f"L{l}: before proj mean {p0.mean():+.3f} std {p0.std():.3f} | after: mean {p1.mean():+.4f} "
          f"max|p-mu| {float((p1-0.3).abs().max()):.4f} rel {float(rel):.2e} | ||h|| median {float(hn.median()):.1f}")
    ok &= float(rel) < 1e-2
print("logits changed under hooks:", float((hooked - ref).abs().max()) > 0,
      "| restored exactly after disable:", bool(t.equal(back, ref)))
ok &= t.equal(back, ref)
abl.enable(); m, _ = H.eval_split(model, H.dataset(1.0).test); abl.disable()
print(f"base OOD under random-direction mean-ablation: {m['acc_2way']:.4f} (ref {H.BASE_OOD_REF})")
print("SMOKE_OK" if ok else "SMOKE_FAIL")
